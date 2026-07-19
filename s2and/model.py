from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias, TypeVar, cast

import lightgbm as lgb
import numpy as np
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.base import clone
from tqdm import tqdm

from s2and import memory_budget
from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    ValidatedArrowInputs,
    normalize_arrow_paths,
    validate_arrow_prediction_artifacts,
)
from s2and.consts import (
    DEFAULT_CHUNK_SIZE,
    LARGE_DISTANCE,
    LARGE_INTEGER,
    NORMALIZATION_VERSION,
)
from s2and.data import ANDData, _load_name_tuples_from_file
from s2and.eval import b3_precision_recall_fscore
from s2and.feature_port import (
    _get_rust_featurizer,
    build_rust_featurizer_from_arrow_paths,
    update_rust_cluster_seeds,
)
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.incremental_linking.feature_block import (
    cluster_seed_disallows_path_from_arrow_paths,
    normalize_cluster_seed_disallow_pairs,
    read_cluster_seed_disallows_arrow,
)
from s2and.incremental_linking.feature_block import (
    read_altered_cluster_signatures_arrow as _read_altered_cluster_signatures_arrow_file,
)
from s2and.incremental_linking.feature_block import (
    read_cluster_seeds_arrow as _read_cluster_seeds_arrow_file,
)
from s2and.incremental_linking.feature_block_arrow import read_arrow_batch_lookup_index_batch_indices_for_request
from s2and.incremental_linking.policy import (
    clusterer_uses_embedding_features,
    clusterer_uses_name_count_features,
    request_cluster_seed_disallow_parts,
    require_dataset_name_counts_binding_for_clusterer,
    require_rust_featurizer_name_counts_binding_for_clusterer,
    resolve_load_name_counts_policy,
)
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer as _require_arrow_name_counts_index_for_clusterer,
)
from s2and.incremental_linking.production import predict_incremental_promoted_linker_from_arrow_paths
from s2and.model_pairwise import FastCluster, intify, predict_pairwise_class0
from s2and.model_pairwise import PairwiseModeler as PairwiseModeler
from s2and.name_counts_index import validated_name_counts_provenance
from s2and.runtime import (
    RuntimeContext,
    build_runtime_context,
    dataset_stage_uses_rust,
    stage_uses_rust,
)
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    get_constraints_matrix_indexed_rust,
)
from s2and.subblocking import (
    GraphSubblockingConfig,
    _resolve_graph_subblocking_config,
    cluster_with_specter,
    make_arrow_graph_subblocking_cluster_fn,
    make_dataset_graph_subblocking_cluster_fn,
    make_subblocks,
)
from s2and.text import (
    canonical_name_tuple_pair,
    canonicalize_name_parts,
    first_names_name_compatible,
    normalize_orcid_compact,
)
from s2and.thread_config import resolve_n_jobs

logger = logging.getLogger("s2and")
IncrementalPhaseBMode = Literal["exact"]
IncrementalBroadcastMode = Literal["always", "never", "top1_consensus"]
IncrementalSeedScoreMode = Literal["mean", "min", "mean_min_hybrid"]
IncrementalDistStats = tuple[float, int, float]
_TReturn = TypeVar("_TReturn")
_MISSING = object()
_ALTERED_PRESPLIT_CACHE_MAX_ENTRIES = 128
_PATH_CACHE_KEY: TypeAlias = tuple[str, int | None, int | None, Any]


@dataclass(frozen=True)
class _ArrowIncrementalSignatureInfo:
    """Signature fields needed by Arrow-only incremental completion."""

    paper_id: str | None
    author_info_first: str | None
    author_info_last: str | None
    author_info_first_normalized_without_apostrophe: str | None
    author_info_orcid: str | None


class _ArrowIncrementalPredictionDataset:
    """Request-state adapter for direct Arrow incremental prediction."""

    def __init__(
        self,
        *,
        arrow_paths: Mapping[str, Any],
        name_tuples: set[tuple[str, str]] | str | None,
        cluster_seeds_require: Mapping[Any, Any] | None,
        cluster_seeds_source: Literal["dataset", "arrow"],
        cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None,
        altered_cluster_signatures: Sequence[Any] | None,
        max_seed_cluster_id: int,
        signatures: Mapping[str, _ArrowIncrementalSignatureInfo],
    ) -> None:
        self.arrow_paths = dict(arrow_paths)
        self.name_tuples = _name_tuples_for_incremental_rules(name_tuples)
        self.cluster_seeds_require = _normalize_cluster_seeds_require(cluster_seeds_require or {})
        self._cluster_seeds_source = cluster_seeds_source
        self.cluster_seeds_disallow = (
            normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow)
            if cluster_seeds_disallow is not None
            else set()
        )
        self.altered_cluster_signatures = (
            [str(signature_id) for signature_id in altered_cluster_signatures]
            if altered_cluster_signatures is not None
            else None
        )
        self.max_seed_cluster_id = int(max_seed_cluster_id)
        self.signatures = dict(signatures)


def _is_recoverable_graph_subblocking_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    exc_type = type(exc)
    return exc_type.__module__.split(".", maxsplit=1)[0] == "pyarrow" or exc_type.__name__.startswith("Arrow")


class _GraphSubblockingFallbackWithLegacyFallback:
    """Call graph subblocking first, then fall back to the legacy SPECTER path."""

    def __init__(
        self,
        graph_fallback: Callable[..., dict[str, list[str]]],
        *,
        source: str,
        candidate_signature_count: int = 0,
    ) -> None:
        self.graph_fallback = graph_fallback
        self.source = source
        self.telemetry: dict[str, Any] = {
            "enabled": 1,
            "mode": "graph",
            "source": source,
            "candidate_signature_count": int(candidate_signature_count),
            "legacy_fallback_invocation_count": 0,
            "graph_prepare_failed": 0,
            "graph_prepare_error": None,
            "graph_fallback_errors": [],
        }

    @property
    def legacy_fallback_invocation_count(self) -> int:
        return int(self.telemetry["legacy_fallback_invocation_count"])

    @property
    def graph_prepare_failed(self) -> bool:
        return bool(self.telemetry["graph_prepare_failed"])

    @property
    def graph_prepare_error(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.telemetry["graph_prepare_error"])

    @property
    def graph_fallback_errors(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.telemetry["graph_fallback_errors"])

    @property
    def stats(self) -> list[dict[str, Any]]:
        return list(getattr(self.graph_fallback, "stats", []) or [])

    @property
    def load_seconds(self) -> float:
        return float(getattr(self.graph_fallback, "load_seconds", 0.0) or 0.0)

    @property
    def load_metrics(self) -> dict[str, Any]:
        return dict(getattr(self.graph_fallback, "load_metrics", {}) or {})

    def prepare(self, signature_groups: Sequence[Sequence[str]]) -> None:
        prepare_graph = getattr(self.graph_fallback, "prepare", None)
        if not callable(prepare_graph):
            return
        group_count = 0
        signature_count = 0
        prepared_groups: list[tuple[str, ...]] = []
        for group in signature_groups:
            prepared_group = tuple(str(signature_id) for signature_id in group)
            prepared_groups.append(prepared_group)
            group_count += 1
            signature_count += len(prepared_group)
        try:
            prepare_graph(prepared_groups)
        except Exception as exc:
            if not _is_recoverable_graph_subblocking_error(exc):
                raise
            if self.source == "arrow":
                raise
            self.telemetry["graph_prepare_failed"] = 1
            self.telemetry["graph_prepare_error"] = self._error_payload(
                exc,
                stage="prepare",
                group_count=group_count,
                signature_count=signature_count,
            )
            logger.warning(
                "Graph subblocking prepare failed; using legacy SPECTER fallback "
                "for graph subblocking fallback calls: source=%s groups=%d signatures=%d",
                self.source,
                group_count,
                signature_count,
                exc_info=True,
            )

    def __call__(
        self,
        signature_ids: Sequence[str],
        anddata: ANDData,
        target_subblock_size: int = 10000,
        **kwargs: Any,
    ) -> dict[str, list[str]]:
        signature_id_list = [str(signature_id) for signature_id in signature_ids]
        if self.graph_prepare_failed:
            return self._call_legacy(signature_id_list, anddata, target_subblock_size, **kwargs)
        try:
            return self.graph_fallback(
                signature_id_list,
                anddata,
                target_subblock_size=target_subblock_size,
                **kwargs,
            )
        except Exception as exc:
            if not _is_recoverable_graph_subblocking_error(exc):
                raise
            if self.source == "arrow":
                raise
            self.graph_fallback_errors.append(
                self._error_payload(exc, stage="call", signature_count=len(signature_id_list))
            )
            logger.warning(
                "Graph subblocking fallback failed; using legacy SPECTER fallback: "
                "source=%s signatures=%d target_subblock_size=%d",
                self.source,
                len(signature_id_list),
                int(target_subblock_size),
                exc_info=True,
            )
            return self._call_legacy(signature_id_list, anddata, target_subblock_size, **kwargs)

    def _call_legacy(
        self,
        signature_ids: list[str],
        anddata: ANDData,
        target_subblock_size: int,
        **kwargs: Any,
    ) -> dict[str, list[str]]:
        self.telemetry["legacy_fallback_invocation_count"] = self.legacy_fallback_invocation_count + 1
        return cluster_with_specter(
            signature_ids,
            anddata,
            target_subblock_size=target_subblock_size,
            **kwargs,
        )

    @staticmethod
    def _error_payload(
        exc: Exception,
        *,
        stage: str,
        signature_count: int,
        group_count: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "signature_count": int(signature_count),
        }
        if group_count is not None:
            payload["group_count"] = int(group_count)
        return payload


def _build_incremental_result(
    clusters: dict[str, list[str]],
    *,
    phase_b_mode: IncrementalPhaseBMode,
    phase_b_budget_bytes: int,
    phase_b_required_bytes: int,
    phase_b_residual_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "clusters": clusters,
        "phase_b_mode": phase_b_mode,
        "phase_b_budget_bytes": int(phase_b_budget_bytes),
        "phase_b_required_bytes": int(phase_b_required_bytes),
    }
    if phase_b_residual_count is not None:
        result["phase_b_residual_count"] = int(phase_b_residual_count)
    return result


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


def _path_full_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        while chunk := infile.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_cache_fingerprint(path_value: Any, *, hash_content: bool = True) -> _PATH_CACHE_KEY:
    path = Path(str(path_value))
    if not hash_content:
        try:
            stat = path.stat()
        except OSError:
            return str(path_value), None, None, None
        return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns), None

    for _attempt in range(3):
        try:
            before = path.stat()
            digest = _path_full_digest(path)
            after = path.stat()
        except OSError:
            return str(path_value), None, None, None
        if (int(before.st_size), int(before.st_mtime_ns)) == (
            int(after.st_size),
            int(after.st_mtime_ns),
        ):
            return str(path.resolve()), int(after.st_size), int(after.st_mtime_ns), digest
    # A concurrently rewritten sidecar must never hit a cache entry. The
    # monotonic token deliberately makes this read uncached while its writer
    # is active; the next stable read regains normal cache reuse.
    return str(path.resolve()), int(after.st_size), int(after.st_mtime_ns), ("unstable", time.monotonic_ns())


def _arrow_paths_cache_fingerprint(arrow_paths: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if arrow_paths is None:
        return ()
    return tuple(
        sorted(
            (
                str(key),
                _path_cache_fingerprint(
                    value,
                    hash_content=str(key) in {"cluster_seed_disallows", "altered_cluster_signatures"},
                ),
            )
            for key, value in arrow_paths.items()
            if value is not None and str(key) != "cluster_seeds"
        )
    )


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
    state = getattr(estimator, "__dict__", None)
    if isinstance(state, Mapping):
        return (type(estimator).__module__, type(estimator).__qualname__, _cacheable_value(state))
    return (type(estimator).__module__, type(estimator).__qualname__)


def _altered_presplit_cache(clusterer: Any) -> OrderedDict[tuple[Any, ...], tuple[tuple[str, ...], ...]]:
    cache = getattr(clusterer, "_s2and_altered_presplit_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        clusterer._s2and_altered_presplit_cache = cache
    return cache


def _get_altered_presplit_cache_entry(
    clusterer: Any,
    key: tuple[Any, ...],
) -> tuple[tuple[str, ...], ...] | None:
    cache = _altered_presplit_cache(clusterer)
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
    cache = _altered_presplit_cache(clusterer)
    cache[key] = tuple(tuple(str(signature_id) for signature_id in cluster) for cluster in clusters)
    cache.move_to_end(key)
    while len(cache) > _ALTERED_PRESPLIT_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def _altered_presplit_cache_key(
    *,
    mode: str,
    altered_cluster_num: int | str,
    signature_ids: Sequence[str],
    effective_partial_supervision: Mapping[tuple[Any, Any], int | float],
    arrow_paths: Mapping[str, Any] | None,
    name_tuples: Any,
    model_fingerprint: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (
        "altered_presplit_v1",
        str(mode),
        str(altered_cluster_num),
        tuple(str(signature_id) for signature_id in signature_ids),
        _partial_supervision_cache_fingerprint(effective_partial_supervision),
        _arrow_paths_cache_fingerprint(arrow_paths),
        _cacheable_value(name_tuples),
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
        _cluster_seed_disallows_for_request(dataset, None) if cluster_seed_disallows is None else cluster_seed_disallows
    )
    if _has_in_profile_pair(disallow_pairs, signature_ids):
        return False
    orcid_values = [_normalized_orcid_for_presplit_skip(dataset, str(signature_id)) for signature_id in signature_ids]
    if any(orcid is None for orcid in orcid_values):
        return False
    return len(set(orcid_values)) == 1


def _count_selected_features(featurizer_info: FeaturizationInfo) -> int:
    """Count the number of feature indices selected by features_to_use."""
    return len(_selected_feature_indices(featurizer_info))


def _read_cluster_seeds_arrow(path: Path) -> dict[str, str]:
    """Read cluster seeds for the current request."""

    return dict(_read_cluster_seeds_arrow_file(path))


def _cluster_seeds_require_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> dict[str, str]:
    if arrow_paths is None:
        return {}
    path_value = arrow_paths.get("cluster_seeds")
    if path_value is None:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Arrow path cluster_seeds={path} does not exist")
    return _read_cluster_seeds_arrow(path)


def _seed_cluster_count_from_seed_map(cluster_seeds_require: Mapping[Any, Any]) -> int:
    """Return the number of unique seed components in the supplied seed map."""

    return len({str(cluster_id) for cluster_id in cluster_seeds_require.values()})


def _optional_arrow_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _signature_batch_indices_for_request(
    arrow_paths: Mapping[str, Any],
    signatures_path: Path,
    requested_signature_ids: Sequence[str],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            read_arrow_batch_lookup_index_batch_indices_for_request(
                signatures_path,
                Path(str(arrow_paths["signatures_batch_index"])),
                key_column="signature_id",
                values=requested_signature_ids,
            )
        )
    )


def _load_arrow_incremental_signature_info(
    arrow_paths: Mapping[str, Any],
    signature_ids: Iterable[Any],
) -> dict[str, _ArrowIncrementalSignatureInfo]:
    """Load request-local signature metadata from `signatures.arrow`."""

    requested_signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
    if not requested_signature_ids:
        return {}
    requested_signature_id_set = set(requested_signature_ids)
    signatures_path = Path(str(arrow_paths["signatures"]))
    required_columns = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
    }
    import pyarrow as pa

    pc: Any = __import__("pyarrow.compute").compute

    value_set = pa.array(requested_signature_ids, type=pa.string())
    signatures: dict[str, _ArrowIncrementalSignatureInfo] = {}
    with pa.memory_map(str(signatures_path), "r") as source:
        reader = pa.ipc.open_file(source)
        missing_columns = sorted(required_columns.difference(reader.schema.names))
        if missing_columns:
            raise ValueError(
                "signatures Arrow is missing required columns for incremental metadata: " f"{missing_columns}"
            )
        selected_columns = required_columns.union({"author_orcid"}.intersection(reader.schema.names))
        required_column_names = sorted(selected_columns)
        signature_id_column_index = reader.schema.get_field_index("signature_id")
        for batch_index in _signature_batch_indices_for_request(arrow_paths, signatures_path, requested_signature_ids):
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
                first_without_apostrophe = canonicalize_name_parts(first, middle, None).first
                signatures[signature_id] = _ArrowIncrementalSignatureInfo(
                    paper_id=_optional_arrow_string(row.get("paper_id")),
                    author_info_first=first,
                    author_info_last=_optional_arrow_string(row.get("author_last")),
                    author_info_first_normalized_without_apostrophe=first_without_apostrophe or None,
                    author_info_orcid=_optional_arrow_string(row.get("author_orcid")),
                )
    missing_signature_ids = [signature_id for signature_id in requested_signature_ids if signature_id not in signatures]
    if missing_signature_ids:
        raise ValueError(f"Arrow signatures are missing incremental signature ids: {missing_signature_ids[:10]}")
    return signatures


def _load_arrow_incremental_orcid_signature_info(
    arrow_paths: Mapping[str, Any],
    signature_ids: Iterable[Any],
) -> dict[str, _ArrowIncrementalSignatureInfo]:
    """Load ORCID-only seed metadata for promoted incremental budget sizing.

    Missing seed signatures are skipped to match ANDData's missing-signature
    ORCID behavior.
    """

    requested_signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
    if not requested_signature_ids:
        return {}
    requested_signature_id_set = set(requested_signature_ids)
    signatures_path = Path(str(arrow_paths["signatures"]))
    required_columns = {"signature_id"}
    import pyarrow as pa

    pc: Any = __import__("pyarrow.compute").compute

    value_set = pa.array(requested_signature_ids, type=pa.string())
    signatures: dict[str, _ArrowIncrementalSignatureInfo] = {}
    with pa.memory_map(str(signatures_path), "r") as source:
        reader = pa.ipc.open_file(source)
        missing_columns = sorted(required_columns.difference(reader.schema.names))
        if missing_columns:
            raise ValueError(
                "signatures Arrow is missing required columns for incremental ORCID metadata: " f"{missing_columns}"
            )
        if "author_orcid" not in reader.schema.names:
            return {}
        required_column_names = ["author_orcid", "signature_id"]
        signature_id_column_index = reader.schema.get_field_index("signature_id")
        for batch_index in _signature_batch_indices_for_request(arrow_paths, signatures_path, requested_signature_ids):
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
                signatures[signature_id] = _ArrowIncrementalSignatureInfo(
                    paper_id=None,
                    author_info_first=None,
                    author_info_last=None,
                    author_info_first_normalized_without_apostrophe=None,
                    author_info_orcid=_optional_arrow_string(row.get("author_orcid")),
                )
    return signatures


def _cluster_seeds_arrow_path_exists(arrow_paths: Mapping[str, Any] | None) -> bool:
    if arrow_paths is None:
        return False
    path_value = arrow_paths.get("cluster_seeds")
    return path_value is not None and Path(str(path_value)).exists()


def _has_incremental_seed_source(dataset: Any, arrow_paths: Mapping[str, Any] | None) -> bool:
    return bool(getattr(dataset, "cluster_seeds_require", {}) or {}) or _cluster_seeds_arrow_path_exists(arrow_paths)


def _require_incremental_seed_source(
    dataset: Any,
    arrow_paths: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    if _has_incremental_seed_source(dataset, arrow_paths):
        return
    missing_files: dict[str, str] = {}
    if arrow_paths is not None and arrow_paths.get("cluster_seeds") is not None:
        missing_files["cluster_seeds"] = str(arrow_paths["cluster_seeds"])
    raise MissingArrowArtifactError(
        context=context,
        required_keys=("cluster_seeds_source",),
        missing_keys=("cluster_seeds_source",),
        missing_files=missing_files,
        producer_hint=(
            "pass seed assignments through dataset.cluster_seeds_require or include a valid "
            "cluster_seeds.arrow in the Arrow path mapping; promoted incremental Rust prediction "
            "does not infer an empty seed source"
        ),
    )


def _missing_arrow_prediction_artifacts_error(
    clusterer: Any,
    *,
    context: str,
    producer_hint: str,
    arrow_paths: Mapping[str, Any] | None = None,
) -> MissingArrowArtifactError:
    required = ["signatures", "papers", "paper_authors"]
    if clusterer_uses_embedding_features(clusterer):
        required.append("specter")
    if clusterer_uses_name_count_features(clusterer):
        required.append("name_counts_index")
    if arrow_paths is not None:
        try:
            validate_arrow_prediction_artifacts(
                arrow_paths,
                require_specter=clusterer_uses_embedding_features(clusterer),
                require_name_counts_index=clusterer_uses_name_count_features(clusterer),
                expected_normalization_version=_resolve_clusterer_normalization_version(clusterer),
                context=context,
                producer_hint=producer_hint,
            )
        except MissingArrowArtifactError as exc:
            return exc
    return MissingArrowArtifactError(
        context=context,
        required_keys=required,
        missing_keys=required,
        missing_files={},
        producer_hint=producer_hint,
    )


def _normalize_cluster_seeds_require(cluster_seeds_require: Mapping[Any, Any]) -> dict[str, str]:
    """Return the seed map with the same string policy used by Arrow sidecars."""

    return {str(signature_id): str(cluster_id) for signature_id, cluster_id in cluster_seeds_require.items()}


def _cluster_seed_maps_match(left: Mapping[Any, Any], right: Mapping[Any, Any]) -> bool:
    return _normalize_cluster_seeds_require(left) == _normalize_cluster_seeds_require(right)


def _arrow_paths_need_current_cluster_seeds(dataset: Any, arrow_paths: Mapping[str, Any]) -> bool:
    current_cluster_seeds = getattr(dataset, "cluster_seeds_require", {}) or {}
    cluster_seeds_path = arrow_paths.get("cluster_seeds")
    if cluster_seeds_path is None:
        return bool(current_cluster_seeds)
    if not Path(str(cluster_seeds_path)).exists():
        return bool(current_cluster_seeds)
    return not _cluster_seed_maps_match(
        _cluster_seeds_require_from_arrow_paths(arrow_paths),
        current_cluster_seeds,
    )


def _cluster_seeds_require_inverse(
    cluster_seeds_require: Mapping[Any, Any],
) -> dict[int | str, list[str]]:
    inverse: dict[int | str, list[str]] = defaultdict(list)
    for signature_id, cluster_num in cluster_seeds_require.items():
        inverse[str(cluster_num)].append(str(signature_id))
    return inverse


def _read_cluster_seed_disallows_arrow(path: Path) -> set[tuple[str, str]]:
    """Read seed disallow pairs for the current request."""

    return set(read_cluster_seed_disallows_arrow(path))


def _cluster_seed_disallows_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    path = cluster_seed_disallows_path_from_arrow_paths(arrow_paths)
    if path is None:
        return set()
    return _read_cluster_seed_disallows_arrow(path)


def _cluster_seed_disallows_for_request(
    dataset: Any,
    arrow_paths: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    request_disallows, _dataset_disallows, _arrow_disallows = request_cluster_seed_disallow_parts(
        dataset,
        _cluster_seed_disallows_from_arrow_paths(arrow_paths),
    )
    return request_disallows


def _partial_supervision_with_cluster_seed_disallows(
    signatures: Sequence[str],
    dataset: Any,
    partial_supervision: Mapping[tuple[str, str], int | float],
    arrow_paths: Mapping[str, Any] | None = None,
    cluster_seed_disallows: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], int | float]:
    merged: dict[tuple[str, str], int | float] = dict(partial_supervision)
    signature_set = {str(signature_id) for signature_id in signatures}
    disallow_pairs = (
        _cluster_seed_disallows_for_request(dataset, arrow_paths)
        if cluster_seed_disallows is None
        else cluster_seed_disallows
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


def _read_nonempty_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_altered_cluster_signatures_arrow(path: Path) -> list[str]:
    return list(_read_altered_cluster_signatures_arrow_file(path))


def _read_altered_cluster_signatures_file(path: Path) -> list[str]:
    """Read legacy text or Arrow altered-profile sidecars for non-production callers."""

    if path.suffix.lower() == ".arrow":
        return _read_altered_cluster_signatures_arrow(path)
    return _read_nonempty_text_lines(path)


def _dataset_altered_cluster_signatures(
    dataset: Any,
    arrow_paths: Mapping[str, Any] | None = None,
) -> list[str]:
    return _altered_cluster_signatures_from_values_or_arrow(
        getattr(dataset, "altered_cluster_signatures", None),
        arrow_paths,
    )


def _altered_cluster_signatures_from_values_or_arrow(
    values: Sequence[Any] | None,
    arrow_paths: Mapping[str, Any] | None = None,
) -> list[str]:
    if values is not None:
        return [str(value) for value in values]
    if arrow_paths is not None:
        path_value = arrow_paths.get("altered_cluster_signatures")
        if path_value is not None:
            path = Path(str(path_value))
            if path.suffix.lower() != ".arrow":
                raise ValueError(
                    "Arrow production altered_cluster_signatures sidecars must use "
                    "altered_cluster_signatures.arrow; text files are only supported by legacy ANDData/training inputs"
                )
            if not path.exists():
                raise MissingArrowArtifactError(
                    context="altered cluster signatures",
                    required_keys=("altered_cluster_signatures",),
                    missing_keys=(),
                    missing_files={"altered_cluster_signatures": str(path)},
                    producer_hint=(
                        "omit altered_cluster_signatures when no altered claimed profiles are present, "
                        "or generate a valid altered_cluster_signatures.arrow sidecar"
                    ),
                )
            return _read_altered_cluster_signatures_file(path)
    return []


def _selected_feature_indices(featurizer_info: FeaturizationInfo) -> list[int]:
    indices: set[int] = set()
    for feature_name in featurizer_info.features_to_use:
        indices.update(featurizer_info.feature_group_to_index[feature_name])
    return sorted(indices)


def _condensed_pair_index(block_size: int, left: int, right: int) -> int:
    if left >= right:
        raise ValueError(f"Expected left < right; got left={left} right={right}")
    return int(block_size * left - (left * (left + 1) // 2) + (right - left - 1))


def _build_partial_supervision_offset_maps_for_block(
    signatures: list[str],
    partial_supervision: dict[tuple[str, str], int | float],
) -> tuple[dict[int, float], dict[int, float]]:
    if not partial_supervision:
        return {}, {}
    signature_to_local_idx = {signature: idx for idx, signature in enumerate(signatures)}
    block_size = len(signatures)
    direct_overrides: dict[int, float] = {}
    reverse_overrides: dict[int, float] = {}
    for (sig_id_1, sig_id_2), value in partial_supervision.items():
        left = signature_to_local_idx.get(sig_id_1)
        right = signature_to_local_idx.get(sig_id_2)
        if left is None or right is None or left == right:
            continue
        adjusted = float(value - LARGE_INTEGER)
        if left < right:
            offset = _condensed_pair_index(block_size, left, right)
            direct_overrides[offset] = adjusted
        else:
            offset = _condensed_pair_index(block_size, right, left)
            reverse_overrides[offset] = adjusted
    return direct_overrides, reverse_overrides


def _compute_predict_batch_chunk_plan(
    num_features: int,
    *,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_pairs: int,
    total_ram_bytes: int | None = None,
) -> memory_budget.RustBatchChunkPlan | None:
    """Compute a bounded pair-batch plan for exact/non-incremental prediction."""
    if total_ram_bytes is None:
        return None
    return memory_budget.compute_rust_batch_chunk_plan(
        num_features=num_features,
        total_pairs=max(1, int(total_pairs)),
        total_rows=max(1, int(total_pairs)),
        selected_feature_count=selected_feature_count,
        nameless_feature_count=nameless_feature_count,
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )


def _predict_distance_matrix_bytes(
    block_size: int,
    *,
    uses_fastcluster: bool,
    fastcluster_fused_dtype: Any,
) -> int:
    """Estimate bytes needed for the per-block distance matrix allocation."""
    if block_size <= 1:
        return 0
    if uses_fastcluster:
        itemsize = int(np.dtype(fastcluster_fused_dtype).itemsize)
        return int(block_size * (block_size - 1) // 2 * itemsize)
    return int(block_size * block_size * np.dtype(np.float16).itemsize)


def _guard_predict_block_matrix_allocation(
    *,
    block_key: str,
    block_size: int,
    uses_fastcluster: bool,
    fastcluster_fused_dtype: Any,
    total_ram_bytes: int | None,
) -> None:
    """Fail fast before exact prediction allocates a block matrix above the RAM budget."""
    if total_ram_bytes is None or block_size <= 1:
        return
    snapshot = memory_budget.memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )
    matrix_bytes = _predict_distance_matrix_bytes(
        block_size,
        uses_fastcluster=uses_fastcluster,
        fastcluster_fused_dtype=fastcluster_fused_dtype,
    )
    if matrix_bytes <= int(snapshot.available_bytes):
        return
    raise MemoryError(
        "Predict exact block exceeds memory budget before matrix allocation: "
        f"block={block_key} block_size={block_size} matrix_bytes={matrix_bytes} "
        f"available_bytes={int(snapshot.available_bytes)} total_ram_bytes={int(snapshot.total_ram_bytes)} "
        f"current_rss_bytes={int(snapshot.current_rss_bytes)} total_ram_source={snapshot.total_ram_source} "
        f"current_rss_source={snapshot.current_rss_source}"
    )


def _signature_first_for_rules(signature: Any) -> str:
    return signature.author_info_first_normalized_without_apostrophe or signature.author_info_first or ""


@lru_cache(maxsize=1)
def _load_name_tuples_for_incremental_rules() -> frozenset[tuple[str, str]]:
    return frozenset(_load_name_tuples_from_file("s2and_name_tuples_canonical.txt"))


def _name_tuples_for_incremental_rules(
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | str | None,
) -> set[tuple[str, str]] | frozenset[tuple[str, str]]:
    if isinstance(name_tuples, frozenset):
        return name_tuples
    if isinstance(name_tuples, set):
        return {canonical_name_tuple_pair(first_a, first_b) for first_a, first_b in name_tuples}
    if name_tuples not in {None, "filtered"}:
        raise ValueError("name_tuples must be None, 'filtered', or a set/frozenset of (first_a, first_b) tuples")
    return _load_name_tuples_for_incremental_rules()


def _required_incremental_linker_artifact_dir(clusterer: Any) -> Path:
    """Return the explicitly attached linker path; never select another bundle implicitly."""

    value = getattr(clusterer, "incremental_linker_artifact_dir", None)
    if value is None:
        value = getattr(getattr(clusterer, "incremental_linker_artifact", None), "artifact_dir", None)
    if value is None:
        raise FileNotFoundError(
            "Promoted incremental prediction requires an attached incremental linker artifact. "
            "Load a complete production bundle or set incremental_linker_artifact_dir explicitly."
        )
    return Path(value)


def _signature_first_initials_for_rules(first: str) -> frozenset[str]:
    tokens = [token for token in first.replace("-", " ").split() if token]
    if not tokens and first:
        stripped = first.strip()
        if stripped:
            tokens = [stripped]
    return frozenset(token[0] for token in tokens if token)


def _residual_phase_b_first_initial_groups(
    clusterer: Any,
    dataset: Any,
    signature_ids: Sequence[str],
    partial_supervision: Mapping[tuple[str, str], int | float],
) -> list[list[str]]:
    """Split residual Phase B by hard first-initial incompatibility when exact parity is safe."""

    residual_signature_ids = [str(signature_id) for signature_id in signature_ids]
    if len(residual_signature_ids) <= 1:
        return [residual_signature_ids]
    if not bool(getattr(clusterer, "use_default_constraints_as_supervision", True)):
        return [residual_signature_ids]
    signatures = getattr(dataset, "signatures", None)
    if not isinstance(signatures, Mapping):
        return [residual_signature_ids]

    initials: dict[str, frozenset[str]] = {}
    for signature_id in residual_signature_ids:
        signature = signatures.get(signature_id)
        if signature is None:
            return [residual_signature_ids]
        first = _signature_first_for_rules(signature)
        if not first:
            return [residual_signature_ids]
        first_initials = _signature_first_initials_for_rules(first)
        if not first_initials:
            return [residual_signature_ids]
        initials[signature_id] = first_initials
    if len(set().union(*initials.values())) <= 1:
        return [residual_signature_ids]

    parent = {signature_id: signature_id for signature_id in residual_signature_ids}

    def find(signature_id: str) -> str:
        root = signature_id
        while parent[root] != root:
            root = parent[root]
        while parent[signature_id] != signature_id:
            next_signature_id = parent[signature_id]
            parent[signature_id] = root
            signature_id = next_signature_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    initial_representatives: dict[str, str] = {}
    for signature_id in residual_signature_ids:
        for initial in initials[signature_id]:
            representative = initial_representatives.setdefault(initial, signature_id)
            union(representative, signature_id)

    if not bool(getattr(clusterer, "suppress_orcid", False)):
        orcid_representatives: dict[str, str] = {}
        for signature_id in residual_signature_ids:
            orcid = _normalized_orcid_for_presplit_skip(dataset, signature_id)
            if orcid is None:
                continue
            representative = orcid_representatives.setdefault(orcid, signature_id)
            union(representative, signature_id)

    residual_signature_id_set = set(residual_signature_ids)
    for (left, right), value in partial_supervision.items():
        left_id = str(left)
        right_id = str(right)
        if left_id not in residual_signature_id_set or right_id not in residual_signature_id_set:
            continue
        if float(value) < LARGE_DISTANCE:
            union(left_id, right_id)

    groups_by_root: dict[str, list[str]] = defaultdict(list)
    for signature_id in residual_signature_ids:
        groups_by_root[find(signature_id)].append(signature_id)
    groups = list(groups_by_root.values())
    if len(groups) <= 1:
        return [residual_signature_ids]
    return groups


def _next_unused_cluster_id(pred_clusters: dict[str, Any], start: int) -> int:
    cluster_id = int(start)
    while str(cluster_id) in pred_clusters:
        cluster_id += 1
    return cluster_id


def _ensure_lightgbm_fitted(clf: Any) -> None:
    if clf is None:
        return
    inner = getattr(clf, "classifier", None)
    if inner is not None and inner is not clf:
        _ensure_lightgbm_fitted(inner)
    if not hasattr(lgb, "LGBMModel") or not isinstance(clf, lgb.LGBMModel):
        return
    booster = getattr(clf, "_Booster", None)
    if booster is None:
        raise RuntimeError(
            "LightGBM estimator has no fitted booster (_Booster is None); " "fit the estimator before prediction."
        )
    if not getattr(clf, "fitted_", False):
        logger.debug("Patching missing LightGBM fitted_ flag for estimator=%s", type(clf).__name__)
        clf.fitted_ = True
    if not hasattr(clf, "n_features_in_"):
        n_feat = getattr(clf, "_n_features", None)
        if n_feat is not None:
            logger.debug(
                "Patching missing LightGBM n_features_in_ from _n_features=%d for estimator=%s",
                int(n_feat),
                type(clf).__name__,
            )
            clf.n_features_in_ = n_feat


def _propagate_n_jobs(estimator: Any, n_jobs: int) -> None:
    """Best-effort propagation of `n_jobs` into estimators/wrappers.

    Keeps S2AND's `Clusterer.n_jobs` as the single knob for both Rust `num_threads` and
    Python model inference thread pools (LightGBM/OpenMP, sklearn estimators, etc.).
    """
    if estimator is None:
        return

    inner = getattr(estimator, "classifier", None)
    if inner is not None and inner is not estimator:
        _propagate_n_jobs(inner, n_jobs)

    for attr in ("estimators", "estimators_"):
        children = getattr(estimator, attr, None)
        if not isinstance(children, list | tuple):
            continue
        for child in children:
            if isinstance(child, tuple) and len(child) == 2:
                _propagate_n_jobs(child[1], n_jobs)
            else:
                _propagate_n_jobs(child, n_jobs)

    resolved_n_jobs = resolve_n_jobs(n_jobs)
    set_params = getattr(estimator, "set_params", None)
    if callable(set_params):
        try:
            set_params(n_jobs=resolved_n_jobs)
        except TypeError as exc:
            logger.debug(
                "Skipping set_params n_jobs propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via set_params for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise

    if hasattr(estimator, "n_jobs"):
        try:
            estimator.n_jobs = resolved_n_jobs
        except (AttributeError, TypeError) as exc:
            logger.debug(
                "Skipping n_jobs attribute propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via attribute assignment for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise


def _resolve_clusterer_normalization_version(clusterer: Any) -> str:
    """Require the package's canonical normalization policy on a model."""

    contract = getattr(clusterer, "feature_contract", None)
    value = contract.get("normalization_version") if isinstance(contract, Mapping) else None
    if value != NORMALIZATION_VERSION:
        raise ValueError(
            "Clusterer feature_contract requires " f"normalization_version={NORMALIZATION_VERSION!r}, got {value!r}"
        )
    return NORMALIZATION_VERSION


def _predict_class0_with_runtime(
    classifier: Any,
    features: np.ndarray,
    *,
    num_threads: int | None = None,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, float, str]:
    raw_features = np.asarray(features)
    native_backend = str(getattr(classifier, "prediction_backend", "python")) == "rust_lightgbm"
    features_2d = raw_features if native_backend else np.asarray(raw_features, dtype=np.float64, order="C")
    if features_2d.size == 0:
        return np.asarray([], dtype=np.float64), 0.0, "none"

    if native_backend:
        if features_2d.ndim != 2:
            raise ValueError(f"native scorer features must be 2D, got shape={features_2d.shape}")
        if features_2d.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError(f"native scorer features must be float32 or float64, got {features_2d.dtype}")
        if not features_2d.flags.c_contiguous:
            raise ValueError("native scorer features must be C-contiguous")
        scorer_plan = memory_budget.compute_native_scorer_chunk_plan(
            row_count=int(features_2d.shape[0]),
            feature_count=int(features_2d.shape[1]),
            input_itemsize=int(features_2d.dtype.itemsize),
            total_ram_bytes=total_ram_bytes,
        )
        started = time.perf_counter()
        positive = classifier.predict_proba_positive(
            features_2d,
            max_rows_per_chunk=max(1, scorer_plan.chunk_rows),
        )
        logger.info(
            "Telemetry: native_scorer rows=%d features=%d dtype=%s chunk_rows=%d chunks=%d "
            "predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d",
            int(features_2d.shape[0]),
            int(features_2d.shape[1]),
            str(features_2d.dtype),
            int(scorer_plan.chunk_rows),
            int(scorer_plan.chunk_count),
            int(scorer_plan.predicted_peak_delta_bytes),
            int(scorer_plan.predicted_peak_rss_bytes),
        )
        backend = "rust_lightgbm_chunked" if scorer_plan.chunk_count > 1 else "rust_lightgbm"
        return 1.0 - np.asarray(positive, dtype=np.float64), time.perf_counter() - started, backend

    # Estimator threading is configured through propagated n_jobs; predict_proba(num_threads=...)
    # is LightGBM-specific and breaks sklearn-compatible wrappers.
    del num_threads
    python_start = time.perf_counter()
    predictions = predict_pairwise_class0(classifier, features_2d)
    backend = str(getattr(classifier, "prediction_backend", "python"))
    return predictions, time.perf_counter() - python_start, backend


def _predict_and_combine(
    classifier: Any,
    nameless_classifier: Any | None,
    features: np.ndarray,
    labels: np.ndarray,
    nameless_features: np.ndarray | None,
    batch_label: int | str,
    *,
    num_threads: int | None = None,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, float]:
    """Predict with main (and optional nameless) classifier, log telemetry, return (predictions, seconds)."""
    row_count = int(features.shape[0])
    if row_count <= 0:
        return np.asarray([], dtype=np.float64), 0.0

    predict_flag = np.isnan(labels)
    not_predict_flag = ~predict_flag
    predicted_rows = int(np.count_nonzero(predict_flag))
    predictions = np.zeros(row_count)
    seconds = 0.0

    def _predict_rows(
        main_matrix: np.ndarray,
        nameless_matrix: np.ndarray | None,
        *,
        row_total: int,
    ) -> tuple[np.ndarray, float]:
        main_pred, main_sec, main_backend = _predict_class0_with_runtime(
            classifier,
            main_matrix,
            num_threads=num_threads,
            total_ram_bytes=total_ram_bytes,
        )
        if nameless_classifier is not None:
            if nameless_matrix is None:
                raise RuntimeError("nameless_classifier is configured but nameless feature matrix is missing")
            nl_pred, nl_sec, nl_backend = _predict_class0_with_runtime(
                nameless_classifier,
                nameless_matrix,
                num_threads=num_threads,
                total_ram_bytes=total_ram_bytes,
            )
            logger.info(
                "Telemetry: model_predict batch=%s main=%s nameless=%s main_s=%.3f nl_s=%.3f rows=%d",
                batch_label,
                main_backend,
                nl_backend,
                main_sec,
                nl_sec,
                row_total,
            )
            return (main_pred + nl_pred) / 2, main_sec + nl_sec

        logger.info(
            "Telemetry: model_predict batch=%s main=%s main_s=%.3f rows=%d",
            batch_label,
            main_backend,
            main_sec,
            row_total,
        )
        return main_pred, main_sec

    # Boolean indexing creates a large temporary copy (often comparable to the full
    # features matrix) when most rows are predicted. Avoid that peak by predicting
    # on the full matrix in that case and overriding constrained rows afterwards.
    copy_avoid_threshold_bytes = 128 * (1 << 20)
    would_copy_bytes = 0
    if predicted_rows > 0 and predicted_rows < row_count:
        would_copy_bytes += int(predicted_rows) * int(features.shape[1]) * int(features.dtype.itemsize)
        if nameless_classifier is not None and nameless_features is not None:
            would_copy_bytes += (
                int(predicted_rows) * int(nameless_features.shape[1]) * int(nameless_features.dtype.itemsize)
            )

    predict_on_full_matrix = predicted_rows > 0 and (
        predicted_rows == row_count or would_copy_bytes >= int(copy_avoid_threshold_bytes)
    )

    if predicted_rows > 0:
        if predict_on_full_matrix:
            combined_predictions, batch_seconds = _predict_rows(
                features,
                nameless_features,
                row_total=predicted_rows,
            )
            predictions[:] = combined_predictions
            seconds += batch_seconds
        else:
            predict_features = features[predict_flag, :]
            predict_nameless_features = nameless_features[predict_flag, :] if nameless_features is not None else None
            combined_predictions, batch_seconds = _predict_rows(
                predict_features,
                predict_nameless_features,
                row_total=predicted_rows,
            )
            predictions[predict_flag] = combined_predictions
            seconds += batch_seconds

    if np.any(not_predict_flag):
        # Fill rows where we already had constraints/partial supervision.
        # Undo the LARGE_INTEGER offset that was applied when labels were staged.
        # For classifier outputs, index 0 corresponds to p(not the same).
        predictions[not_predict_flag] = labels[not_predict_flag] + LARGE_INTEGER
    return predictions, seconds


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


def _handle_optional_rust_exception(
    runtime_context: RuntimeContext,
    *,
    strict_message: str,
    exc: Exception,
    python_path_warning: str,
    context_fields: tuple[str, ...] = (),
) -> None:
    details = " ".join((*context_fields, f"run_id={runtime_context.run_id}", f"error={exc}"))
    if stage_uses_rust(runtime_context):
        raise RuntimeError(f"{strict_message} ({details})") from exc
    logger.warning("%s: %s", python_path_warning, exc)


def _optional_rust_or_python_path(
    fn: Callable[[], _TReturn],
    python_fn: Callable[[], _TReturn],
    *,
    runtime_context: RuntimeContext,
    label: str,
    context_fields: tuple[str, ...] = (),
    strict_message: str | None = None,
    python_path_warning: str | None = None,
) -> _TReturn:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - native extension optional
        _handle_optional_rust_exception(
            runtime_context,
            strict_message=(strict_message or f"Rust {label} failed in strict rust backend"),
            exc=exc,
            python_path_warning=(
                python_path_warning
                or f"Optional Rust {label} failed while runtime backend is Python; using Python path"
            ),
            context_fields=context_fields,
        )
        return python_fn()


def _cluster_seeds_version(dataset: ANDData) -> int:
    return int(getattr(dataset, "_cluster_seeds_version", 0))


def _bump_cluster_seeds_version(dataset: ANDData) -> int:
    next_version = _cluster_seeds_version(dataset) + 1
    dataset._cluster_seeds_version = next_version
    return next_version


class _VersionedClusterSeedDict(dict[Any, Any]):
    def __init__(self, *args: Any, on_mutation: Callable[[], None] | None = None, **kwargs: Any) -> None:
        self._on_mutation = on_mutation
        super().__init__(*args, **kwargs)

    def set_on_mutation(self, on_mutation: Callable[[], None] | None) -> None:
        self._on_mutation = on_mutation

    def _mark_mutated(self) -> None:
        callback = self._on_mutation
        if callback is not None:
            callback()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._mark_mutated()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self._mark_mutated()

    def clear(self) -> None:
        if self:
            super().clear()
            self._mark_mutated()

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if key in self:
            value = super().pop(key)
            self._mark_mutated()
            return value
        if default is not _MISSING:
            return default
        raise KeyError(key)

    def popitem(self) -> tuple[Any, Any]:
        value = super().popitem()
        self._mark_mutated()
        return value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return self[key]
        super().__setitem__(key, default)
        self._mark_mutated()
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            super().update(*args, **kwargs)
            self._mark_mutated()

    def __ior__(self, value: Any) -> Self:
        super().__ior__(value)
        self._mark_mutated()
        return self


class _VersionedClusterSeedSet(set[tuple[Any, Any]]):
    def __init__(self, *args: Any, on_mutation: Callable[[], None] | None = None) -> None:
        self._on_mutation = on_mutation
        super().__init__(*args)

    def set_on_mutation(self, on_mutation: Callable[[], None] | None) -> None:
        self._on_mutation = on_mutation

    def _mark_mutated(self) -> None:
        callback = self._on_mutation
        if callback is not None:
            callback()

    def add(self, element: tuple[Any, Any]) -> None:
        super().add(element)
        self._mark_mutated()

    def remove(self, element: tuple[Any, Any]) -> None:
        super().remove(element)
        self._mark_mutated()

    def discard(self, element: tuple[Any, Any]) -> None:
        if element in self:
            super().discard(element)
            self._mark_mutated()

    def pop(self) -> tuple[Any, Any]:
        value = super().pop()
        self._mark_mutated()
        return value

    def clear(self) -> None:
        if self:
            super().clear()
            self._mark_mutated()

    def update(self, *others: Any) -> None:
        if others:
            super().update(*others)
            self._mark_mutated()

    def difference_update(self, *others: Any) -> None:
        if others:
            super().difference_update(*others)
            self._mark_mutated()

    def intersection_update(self, *others: Any) -> None:
        if others:
            super().intersection_update(*others)
            self._mark_mutated()

    def symmetric_difference_update(self, other: Any) -> None:
        super().symmetric_difference_update(other)
        self._mark_mutated()

    def __ior__(self, value: Any) -> Self:
        super().__ior__(value)
        self._mark_mutated()
        return self

    def __iand__(self, value: Any) -> Self:
        super().__iand__(value)
        self._mark_mutated()
        return self

    def __isub__(self, value: Any) -> Self:
        super().__isub__(value)
        self._mark_mutated()
        return self

    def __ixor__(self, value: Any) -> Self:
        super().__ixor__(value)
        self._mark_mutated()
        return self


def _ensure_cluster_seed_version_tracking(dataset: ANDData) -> None:
    def _mark_mutated() -> None:
        _bump_cluster_seeds_version(dataset)

    require = getattr(dataset, "cluster_seeds_require", {})
    if require is None:
        require = {}
    if isinstance(require, _VersionedClusterSeedDict):
        require.set_on_mutation(_mark_mutated)
    else:
        dataset.cluster_seeds_require = _VersionedClusterSeedDict(require, on_mutation=_mark_mutated)

    disallow = getattr(dataset, "cluster_seeds_disallow", set())
    if disallow is None:
        disallow = set()
    if isinstance(disallow, _VersionedClusterSeedSet):
        disallow.set_on_mutation(_mark_mutated)
    else:
        dataset.cluster_seeds_disallow = _VersionedClusterSeedSet(disallow, on_mutation=_mark_mutated)


@dataclass(frozen=True)
class _ConstraintPolicy:
    """Resolved hard-constraint behavior for one constraint evaluation boundary."""

    dont_merge_cluster_seeds: bool = True
    incremental_dont_use_cluster_seeds: bool = False
    suppress_orcid: bool = False


def _sync_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: RuntimeContext | None = None,
) -> None:
    if runtime_context is None:
        runtime_context = dataset.runtime_context
    if _use_rust_constraints(runtime_context, dataset):
        # Best-effort instrumentation for subblocking lifecycle overhead.
        # Stored on the dataset to avoid changing return payloads on hot paths.
        dataset._rust_cluster_seeds_sync_calls = int(getattr(dataset, "_rust_cluster_seeds_sync_calls", 0)) + 1

        _ensure_cluster_seed_version_tracking(dataset)
        seed_version = _cluster_seeds_version(dataset)
        require = getattr(dataset, "cluster_seeds_require", {})
        disallow = getattr(dataset, "cluster_seeds_disallow", set())
        require_id = int(id(require))
        disallow_id = int(id(disallow))
        require_len = int(len(require))
        disallow_len = int(len(disallow))

        last_synced = getattr(dataset, "_rust_cluster_seeds_synced_version", None)
        last_require_id = getattr(dataset, "_rust_cluster_seeds_require_id", None)
        last_disallow_id = getattr(dataset, "_rust_cluster_seeds_disallow_id", None)
        last_require_len = getattr(dataset, "_rust_cluster_seeds_require_len", None)
        last_disallow_len = getattr(dataset, "_rust_cluster_seeds_disallow_len", None)
        if (
            last_synced == seed_version
            and last_require_id == require_id
            and last_require_len == require_len
            and last_disallow_id == disallow_id
            and last_disallow_len == disallow_len
        ):
            dataset._rust_cluster_seeds_sync_skipped_unchanged = (
                int(getattr(dataset, "_rust_cluster_seeds_sync_skipped_unchanged", 0)) + 1
            )
            return

        dataset._rust_cluster_seeds_sync_attempted = int(getattr(dataset, "_rust_cluster_seeds_sync_attempted", 0)) + 1

        def _sync() -> None:
            sync_start = time.perf_counter()
            update_rust_cluster_seeds(dataset, runtime_context=runtime_context, bump_version=False)
            sync_seconds = float(time.perf_counter() - sync_start)
            dataset._rust_cluster_seeds_sync_succeeded = (
                int(getattr(dataset, "_rust_cluster_seeds_sync_succeeded", 0)) + 1
            )
            dataset._rust_cluster_seeds_sync_seconds_total = (
                float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_total", 0.0)) + sync_seconds
            )
            dataset._rust_cluster_seeds_sync_seconds_max = max(
                float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_max", 0.0)),
                sync_seconds,
            )
            dataset._rust_cluster_seeds_synced_version = seed_version
            dataset._rust_cluster_seeds_require_id = require_id
            dataset._rust_cluster_seeds_require_len = require_len
            dataset._rust_cluster_seeds_disallow_id = disallow_id
            dataset._rust_cluster_seeds_disallow_len = disallow_len

        _optional_rust_or_python_path(
            fn=_sync,
            python_fn=lambda: None,
            runtime_context=runtime_context,
            label="cluster seed sync",
            strict_message="Rust cluster seed sync failed in strict rust backend",
            python_path_warning=(
                "Optional Rust cluster seed sync failed while runtime backend is Python; using Python seed state"
            ),
        )


def _initialize_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
) -> tuple[object | None, bool | None]:
    if not use_default_constraints_as_supervision:
        return None, None

    use_rust_constraints = _use_rust_constraints(runtime_context, dataset)
    if not use_rust_constraints:
        return None, False

    rust_featurizer = _optional_rust_or_python_path(
        fn=lambda: _get_rust_featurizer(dataset, runtime_context=runtime_context),
        python_fn=lambda: None,
        runtime_context=runtime_context,
        label="constraint featurizer init",
        strict_message="Rust constraint stage requested but Rust featurizer init failed",
        python_path_warning=(
            "Optional Rust featurizer init failed while runtime backend is Python; using Python constraints"
        ),
    )
    if rust_featurizer is None:
        return None, False

    return rust_featurizer, True


def _resolve_constraint_api_mode(
    rust_featurizer: object | None,
    use_rust_constraints: bool | None,
) -> str:
    if not use_rust_constraints or rust_featurizer is None:
        return "python"
    return "indexed"


def _build_signature_index_by_id(rust_featurizer: object) -> dict[str, int]:
    signature_ids = rust_featurizer.signature_ids()  # type: ignore[attr-defined]
    return {str(sig_id): idx for idx, sig_id in enumerate(signature_ids)}


def _build_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
    suppress_orcid: bool = False,
) -> _IncrementalConstraintBackend:
    """Build Phase A constraint backend state once for reuse across subblocks."""
    rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        runtime_context=runtime_context,
    )
    constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    signature_index_by_id: dict[str, int] | None = None
    if constraint_api_mode == "indexed" and rust_featurizer is not None:
        signature_index_by_id = _optional_rust_or_python_path(
            fn=lambda: _build_signature_index_by_id(rust_featurizer),
            python_fn=lambda: None,
            runtime_context=runtime_context,
            label="indexed constraint setup",
            strict_message="Rust indexed constraint setup failed in strict rust backend",
            python_path_warning=(
                "Optional Rust indexed constraint setup failed while runtime backend is Python; "
                "using Python constraints"
            ),
        )
        if signature_index_by_id is None:
            use_rust_constraints = False
            rust_featurizer = None
            constraint_api_mode = "python"
    return _IncrementalConstraintBackend(
        rust_featurizer=rust_featurizer,
        use_rust_constraints=use_rust_constraints,
        constraint_api_mode=constraint_api_mode,
        signature_index_by_id=signature_index_by_id,
        suppress_orcid=suppress_orcid,
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
) -> tuple[list[float], _ConstraintBatchTelemetry]:
    if constraint_backend is not None:
        rust_featurizer = constraint_backend.rust_featurizer
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

        used_python_path_after_optional_rust_failure = False

        def _resolve_values_python_after_optional_rust_failure() -> list[float | None]:
            nonlocal used_python_path_after_optional_rust_failure
            used_python_path_after_optional_rust_failure = True
            return _resolve_values_python()

        values = _optional_rust_or_python_path(
            fn=_resolve_values_rust,
            python_fn=_resolve_values_python_after_optional_rust_failure,
            runtime_context=runtime_context,
            label="batch constraint evaluation",
            strict_message="Rust batch constraint evaluation failed in strict rust backend",
            python_path_warning=(
                "Optional Rust batch constraint evaluation failed while runtime backend is Python; "
                "using Python constraints"
            ),
            context_fields=(f"pairs={len(unresolved_pairs)}",),
        )
        if used_python_path_after_optional_rust_failure:
            telemetry.api_mode = "optional_rust_failed_python"
            telemetry.rust_batch_call_count = 0
        else:
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


@dataclass(frozen=True)
class _DistanceMatrixChunk:
    block_key: str
    block_size: int
    start_offset: int
    index_i: np.ndarray
    index_j: np.ndarray
    pair_ids: list[tuple[str, str]] | None
    labels: np.ndarray
    block_signature_indices: list[int] | None = None

    def signature_pairs(self) -> list[tuple[str, str, float]]:
        if self.pair_ids is None:
            raise RuntimeError("signature_pairs requested for fused Rust chunk without explicit pair ids")
        return [
            (sig_id_1, sig_id_2, float(label))
            for (sig_id_1, sig_id_2), label in zip(self.pair_ids, self.labels, strict=True)
        ]


@dataclass(frozen=True)
class _PredictedDistanceMatrixChunk:
    chunk: _DistanceMatrixChunk
    predictions: np.ndarray
    batch_seconds: float


@dataclass(frozen=True)
class _PredictedDistanceMatrixBatch:
    batch_num: int
    blocks: list[str]
    indices: list[tuple[int, int]]
    predictions: np.ndarray
    batch_seconds: float


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


@dataclass
class _RustFeaturizerPredictDataset:
    """Minimal dataset shape needed after Arrow inputs have entered Rust."""

    cluster_seeds_require: dict[str, int | str]
    cluster_seeds_disallow: set[tuple[str, str]]
    signatures: Mapping[str, Any] = field(default_factory=dict)
    name: str = "rust_featurizer_predict"


@dataclass(frozen=True)
class _RustFeaturizerSignatureRuleMetadata:
    first: str
    author_info_orcid: str | None

    @property
    def author_info_first(self) -> str:
        return self.first

    @property
    def author_info_first_normalized_without_apostrophe(self) -> str:
        return self.first


def _cluster_seeds_require_from_rust_featurizer(rust_featurizer: Any) -> dict[str, int | str]:
    return {
        str(signature_id): str(component_id) for signature_id, component_id in rust_featurizer.cluster_seeds_require()
    }


def _signature_rule_metadata_from_rust_featurizer(
    rust_featurizer: Any,
) -> dict[str, _RustFeaturizerSignatureRuleMetadata]:
    metadata: dict[str, _RustFeaturizerSignatureRuleMetadata] = {}
    for signature_id, first, orcid in rust_featurizer.signature_rule_metadata():
        first_value = "" if first is None else str(first)
        metadata[str(signature_id)] = _RustFeaturizerSignatureRuleMetadata(
            first=first_value,
            author_info_orcid=None if orcid is None else str(orcid),
        )
    return metadata


def _upper_triangle_indices_for_range(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = int(block_size * (block_size - 1) // 2)
    start = int(start_offset)
    if start < 0 or start > total_pairs:
        raise ValueError(f"start_offset must be in [0, {total_pairs}], got {start}")
    count = total_pairs - start if max_pairs is None else min(int(max_pairs), total_pairs - start)
    if count <= 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    linear = np.arange(start, start + count, dtype=np.int64)
    block_size_int = int(block_size)
    discriminant = (2 * block_size_int - 1) ** 2 - 8 * linear
    left = np.floor(((2 * block_size_int - 1) - np.sqrt(discriminant)) / 2).astype(np.int64)
    row_start = left * (2 * block_size_int - left - 1) // 2

    too_high = row_start > linear
    if np.any(too_high):
        left[too_high] -= 1
        row_start = left * (2 * block_size_int - left - 1) // 2
    next_row_start = (left + 1) * (2 * block_size_int - left - 2) // 2
    too_low = next_row_start <= linear
    if np.any(too_low):
        left[too_low] += 1
        row_start = left * (2 * block_size_int - left - 1) // 2

    right = left + 1 + (linear - row_start)
    return left.astype(np.intp, copy=False), right.astype(np.intp, copy=False)


@dataclass(frozen=True)
class _IncrementalExperimentConfig:
    """Experiment-only controls for incremental single-letter assignment."""

    precluster_broadcast_mode: IncrementalBroadcastMode
    seed_score_mode: IncrementalSeedScoreMode
    mean_min_hybrid_weight: float


def _incremental_cluster_score(
    stats: IncrementalDistStats,
    *,
    config: _IncrementalExperimentConfig,
) -> float:
    """Return the assignment score for one query/seed cluster pair."""

    mean_dist, _count, min_dist = stats
    if config.seed_score_mode == "mean":
        return float(mean_dist)
    if config.seed_score_mode == "min":
        return float(min_dist)
    return float(
        (1.0 - float(config.mean_min_hybrid_weight)) * float(mean_dist)
        + float(config.mean_min_hybrid_weight) * float(min_dist)
    )


class Clusterer:
    """
    A wrapper for learning a clusterer

    Args:
        featurizer_info: FeaturizationInfo
            Featurization information
        classifier: sklearn compatible model
            Classifier which uses pairwise features to make a distance matrix
        val_blocks_size: int
            How many blocks to use during hyperparam optimization.
            Defaults to None, which uses all of them.
        cluster_model: sklearn compatible model
            Clusterer model
            Defaults to None, which uses FastCluster with average linking.
        search_space: Dict
            Search space for the hyperpamater optimization.
            Defaults to None, which uses a space appropriate to FastCluster.
        n_iter: int
            Number of hyperparameter evaluations
        n_jobs: int
            Parallelize each clusterer this many ways
        use_cache: bool
            Whether to use the cache when making distance matrices
        use_default_constraints_as_supervision: bool
            Whether to use the default constraints when constructing the distance matrices.
            These are high precision and can save a lot of compute/time.
        random_state: int
            Random state
        nameless_classifier: sklearn compatible model
            A second classifier which uses pairwise features excluding all name information, and
            whose predictions are averaged with the main classifier. Won't be used if None
        nameless_featurizer_info: FeaturizationInfo
            The FeaturizationInfo for the second classifier. Won't be used if None
        dont_merge_cluster_seeds: bool
            whether to enforce "disallow" constraints for signatures in different required seed clusters
        batch_size: int
            batch size for featurization, lower means less memory, but slower
        suppress_orcid: bool
            Whether default constraint resolution should ignore same-ORCID must-link constraints.
    """

    def __init__(
        self,
        featurizer_info: FeaturizationInfo,
        classifier: Any,
        val_blocks_size: int | None = None,
        cluster_model: Any | None = None,
        search_space: dict[str, Any] | None = None,
        n_iter: int = 25,
        n_jobs: int = 16,
        use_cache: bool = False,
        use_default_constraints_as_supervision: bool = True,
        random_state: int = 42,
        nameless_classifier: Any | None = None,
        nameless_featurizer_info: FeaturizationInfo | None = None,
        dont_merge_cluster_seeds: bool = True,
        batch_size: int = 1000000,
        suppress_orcid: bool = False,
    ):
        self.featurizer_info = featurizer_info
        self.nameless_featurizer_info = nameless_featurizer_info
        self.classifier = copy.deepcopy(classifier)
        self.nameless_classifier = copy.deepcopy(nameless_classifier)
        self.val_blocks_size = val_blocks_size
        self.n_iter = n_iter
        self._n_jobs = 1
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_cache = use_cache
        self.use_default_constraints_as_supervision = use_default_constraints_as_supervision
        self.dont_merge_cluster_seeds = dont_merge_cluster_seeds
        self.suppress_orcid = suppress_orcid
        if cluster_model is None:
            self.cluster_model = FastCluster(linkage="average")
        else:
            self.cluster_model = copy.deepcopy(cluster_model)

        if search_space is None:
            self.search_space = {"eps": hp.uniform("eps", 0, 1)}
        else:
            self.search_space = search_space

        # A freshly constructed clusterer is born from this package, so it
        # inherits the package's normalization policy. Loaded bundles carry
        # their own recorded value instead.
        self.feature_contract = {"normalization_version": NORMALIZATION_VERSION}
        self.hyperopt_trials_store: Trials | list[Trials] | None = None
        self.best_params: dict[Any, Any] | None = None
        self.batch_size = batch_size
        self.incremental_precluster_broadcast_mode: IncrementalBroadcastMode = "always"
        self.incremental_seed_score_mode: IncrementalSeedScoreMode = "mean"
        self.incremental_mean_min_hybrid_weight: float = 0.5
        self.incremental_linker_artifact_dir: Path | None = None
        self.incremental_linker_artifact: Any | None = None
        self.production_model_bundle_dir: Path | None = None
        self.production_model_bundle_version: str | None = None
        self.production_model_bundle_status: str | None = None
        self.subblocking_graph_config = GraphSubblockingConfig()

    @property
    def n_jobs(self) -> int:
        return int(getattr(self, "_n_jobs", 1))

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        n_jobs = resolve_n_jobs(value)
        self._n_jobs = n_jobs
        _propagate_n_jobs(getattr(self, "classifier", None), n_jobs)
        _propagate_n_jobs(getattr(self, "nameless_classifier", None), n_jobs)

    @staticmethod
    def filter_blocks(block_dict: dict[str, list[str]], num_to_keep: int | None = None) -> dict[str, list[str]]:
        """
        Filter out blocks of size 1, as they are not useful or train/val

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        num_to_keep: int
            the number of blocks to keep, keeps all if None

        Returns
        -------
        either the loaded json, or the passed in object
        """
        # blocks with only 1 element are useless for train/val
        # and we can only keep as many as is specified
        out_dict = {}
        count = 0
        for block_key, signatures in block_dict.items():
            if len(signatures) > 1:
                out_dict[block_key] = signatures
                count += 1
                # early stopping if we have enough
                if num_to_keep is not None and count == num_to_keep:
                    return out_dict
        return out_dict

    def _incremental_experiment_config(self) -> _IncrementalExperimentConfig:
        """Return validated experiment controls for incremental assignment."""

        raw_broadcast_mode = str(getattr(self, "incremental_precluster_broadcast_mode", "always"))
        raw_seed_score_mode = str(getattr(self, "incremental_seed_score_mode", "mean"))
        raw_hybrid_weight = float(getattr(self, "incremental_mean_min_hybrid_weight", 0.5))

        valid_broadcast_modes: set[str] = {"always", "never", "top1_consensus"}
        if raw_broadcast_mode not in valid_broadcast_modes:
            raise ValueError(
                "Unsupported incremental_precluster_broadcast_mode="
                f"{raw_broadcast_mode!r}; expected one of {sorted(valid_broadcast_modes)}"
            )
        valid_seed_score_modes: set[str] = {"mean", "min", "mean_min_hybrid"}
        if raw_seed_score_mode not in valid_seed_score_modes:
            raise ValueError(
                "Unsupported incremental_seed_score_mode="
                f"{raw_seed_score_mode!r}; expected one of {sorted(valid_seed_score_modes)}"
            )
        if not 0.0 <= raw_hybrid_weight <= 1.0:
            raise ValueError("incremental_mean_min_hybrid_weight must be in [0, 1]; " f"got {raw_hybrid_weight!r}")
        return _IncrementalExperimentConfig(
            precluster_broadcast_mode=raw_broadcast_mode,  # type: ignore[arg-type]
            seed_score_mode=raw_seed_score_mode,  # type: ignore[arg-type]
            mean_min_hybrid_weight=float(raw_hybrid_weight),
        )

    def _best_incremental_cluster(
        self,
        cluster_dists: Mapping[int | str, IncrementalDistStats],
        *,
        config: _IncrementalExperimentConfig,
    ) -> tuple[int | str | None, float, float]:
        """Return the best and second-best seed-cluster scores for one query."""

        best_cluster_id: int | str | None = None
        best_score = float("inf")
        second_best_score = float("inf")
        for cluster_id, stats in cluster_dists.items():
            score = _incremental_cluster_score(stats, config=config)
            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_cluster_id = cluster_id
            elif score < second_best_score:
                second_best_score = score
        return best_cluster_id, float(best_score), float(second_best_score)

    def _resolve_constraint_batch(
        self,
        dataset: ANDData,
        pair_ids: list[tuple[str, str]],
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        *,
        incremental_dont_use_cluster_seeds: bool,
        constraint_backend: _IncrementalConstraintBackend,
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
        )

    def _flush_completed_block(
        self,
        *,
        block_key: str,
        pairwise_proba: np.ndarray | None,
        block_dict: dict[str, list[str]],
        effective_cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
        pred_clusters: defaultdict[str, list[str]],
    ) -> None:
        if block_key == "" or pairwise_proba is None:
            return
        if not isinstance(self.cluster_model, FastCluster):
            pairwise_proba += pairwise_proba.T
            np.fill_diagonal(pairwise_proba, 0)
        labels = self._cluster_one_block_with_logging(
            block_dict[block_key],
            pairwise_proba,
            effective_cluster_model_params,
            dataset,
            all_disallow_signature_ids,
            block_key=block_key,
        )
        for signature, label in zip(block_dict[block_key], labels, strict=True):
            pred_clusters[block_key + "_" + str(label)].append(signature)

    def distance_matrix_helper(
        self,
        block_dict: dict,
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
    ):
        """
        Helper generator function to yield one pair for batch featurization on the fly

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        dataset: ANDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for exact block-matrix allocation checks.

        Returns
        -------
        yields pairs of ((sig id 1, sig id 2, label), index pair into the distance matrix, block key)
        """
        if runtime_context is None:
            runtime_context = dataset.runtime_context
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
        )

        telemetry = _ConstraintTelemetryAccumulator()
        pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))

        for block_key, signatures in block_dict.items():
            pair_batch_ids: list[tuple[str, str]] = []
            index_batch: list[tuple[int, int]] = []
            for i, j in zip(*np.triu_indices(len(signatures), k=1), strict=True):
                pair_batch_ids.append((signatures[i], signatures[j]))
                index_batch.append((i, j))
                if len(pair_batch_ids) >= pair_chunk_size:
                    labels, batch_telemetry = self._resolve_constraint_batch(
                        dataset,
                        pair_batch_ids,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        constraint_backend=constraint_backend,
                    )
                    _accumulate_constraint_telemetry(telemetry, batch_telemetry)
                    for (sig_id_1, sig_id_2), label, (left, right) in zip(
                        pair_batch_ids, labels, index_batch, strict=True
                    ):
                        yield ((sig_id_1, sig_id_2, label), (left, right), block_key)
                    pair_batch_ids = []
                    index_batch = []

            if pair_batch_ids:
                labels, batch_telemetry = self._resolve_constraint_batch(
                    dataset,
                    pair_batch_ids,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    constraint_backend=constraint_backend,
                )
                _accumulate_constraint_telemetry(telemetry, batch_telemetry)
                for (sig_id_1, sig_id_2), label, (left, right) in zip(pair_batch_ids, labels, index_batch, strict=True):
                    yield ((sig_id_1, sig_id_2, label), (left, right), block_key)

        logger.info(
            "Telemetry: constraint_batch stage=distance_matrix total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            telemetry.total_pairs,
            telemetry.partial_supervision_hits,
            telemetry.unresolved_pairs,
            telemetry.rust_batch_call_count,
            telemetry.api_mode_summary,
            telemetry.elapsed_seconds,
            runtime_context.run_id,
        )

    def _yield_non_fused_chunks(
        self,
        *,
        block_key: str,
        signatures: list[str],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        incremental_dont_use_cluster_seeds: bool,
        constraint_backend: _IncrementalConstraintBackend,
        telemetry: _ConstraintTelemetryAccumulator,
        pair_chunk_size: int | None = None,
        start_offset: int = 0,
    ):
        block_size = len(signatures)
        if block_size <= 1:
            return
        block_pair_count = int(block_size * (block_size - 1) / 2)
        pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
        tri_i, tri_j = np.triu_indices(block_size, k=1)
        offset = int(start_offset)
        if offset < 0 or offset > block_pair_count:
            raise ValueError(f"start_offset must be between 0 and {block_pair_count}, got {start_offset}")
        while offset < block_pair_count:
            end = min(offset + pair_chunk_size, block_pair_count)
            i_chunk = tri_i[offset:end]
            j_chunk = tri_j[offset:end]
            pair_batch_ids = [
                (signatures[int(left)], signatures[int(right)]) for left, right in zip(i_chunk, j_chunk, strict=True)
            ]
            labels, batch_telemetry = self._resolve_constraint_batch(
                dataset,
                pair_batch_ids,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                constraint_backend=constraint_backend,
            )
            _accumulate_constraint_telemetry(telemetry, batch_telemetry)
            yield _DistanceMatrixChunk(
                block_key=block_key,
                block_size=block_size,
                start_offset=offset,
                index_i=i_chunk,
                index_j=j_chunk,
                pair_ids=pair_batch_ids,
                labels=np.asarray(labels, dtype=np.float64),
            )
            offset = end

    def _distance_matrix_chunk_helper_rust(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
    ):
        if runtime_context is None:
            runtime_context = build_runtime_context("constraints", backend="rust")
        if not stage_uses_rust(runtime_context):
            raise ValueError("Rust chunk helper is only valid when runtime_context resolves to rust backend")

        use_cache = bool(self.use_cache)
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
        )
        rust_featurizer = constraint_backend.rust_featurizer
        constraint_api_mode = constraint_backend.constraint_api_mode
        signature_index_by_id = constraint_backend.signature_index_by_id

        telemetry = _ConstraintTelemetryAccumulator()
        pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
        used_fused_path = False
        use_fused_block_api = bool(
            self.use_default_constraints_as_supervision
            and not use_cache
            and constraint_api_mode == "indexed"
            and rust_featurizer is not None
            and signature_index_by_id is not None
        )

        for block_key, signatures in block_dict.items():
            block_size = len(signatures)
            if block_size <= 1:
                continue
            block_pair_count = int(block_size * (block_size - 1) / 2)
            if use_fused_block_api and signature_index_by_id is not None and rust_featurizer is not None:
                block_signature_indices = [int(signature_index_by_id[signature]) for signature in signatures]
                direct_overrides, reverse_overrides = _build_partial_supervision_offset_maps_for_block(
                    signatures,
                    partial_supervision,
                )
                offset = 0
                while offset < block_pair_count:
                    chunk_pair_count = int(min(pair_chunk_size, block_pair_count - offset))
                    constraint_start = time.perf_counter()
                    try:
                        local_i, local_j, values = get_constraints_block_upper_triangle_indexed_rust(
                            block_signature_indices,
                            start_offset=offset,
                            max_pairs=chunk_pair_count,
                            dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                            num_threads=self.n_jobs,
                            featurizer=rust_featurizer,
                            suppress_orcid=constraint_backend.suppress_orcid,
                        )
                    except (RuntimeError, ValueError) as exc:
                        _handle_optional_rust_exception(
                            runtime_context,
                            strict_message="Rust fused block constraint evaluation failed in strict rust backend",
                            exc=exc,
                            python_path_warning=(
                                "Optional Rust fused block constraint evaluation failed while runtime backend "
                                "is Python; "
                                "using non-fused chunk path"
                            ),
                            context_fields=(
                                f"block={block_key}",
                                f"start_offset={offset}",
                                f"pairs={chunk_pair_count}",
                            ),
                        )
                        use_fused_block_api = False
                        break
                    constraint_elapsed = float(time.perf_counter() - constraint_start)
                    if len(local_i) != chunk_pair_count or len(local_j) != chunk_pair_count:
                        raise RuntimeError(
                            "Rust fused block constraint API returned mismatched index lengths: "
                            f"expected={chunk_pair_count} left={len(local_i)} right={len(local_j)}"
                        )
                    if len(values) != chunk_pair_count:
                        raise RuntimeError(
                            "Rust fused block constraint API returned mismatched constraint length: "
                            f"expected={chunk_pair_count} got={len(values)}"
                        )

                    labels = np.full(chunk_pair_count, np.nan, dtype=np.float64)
                    partial_hits_chunk = 0
                    unresolved_chunk = 0
                    for row_offset in range(chunk_pair_count):
                        pair_offset = offset + row_offset
                        override = direct_overrides.get(pair_offset)
                        if override is None:
                            override = reverse_overrides.get(pair_offset)
                        if override is not None:
                            labels[row_offset] = float(override)
                            partial_hits_chunk += 1
                            continue
                        unresolved_chunk += 1
                        value = values[row_offset]
                        if value is not None:
                            labels[row_offset] = float(value - LARGE_INTEGER)

                    telemetry.total_pairs += int(chunk_pair_count)
                    telemetry.partial_supervision_hits += int(partial_hits_chunk)
                    telemetry.unresolved_pairs += int(unresolved_chunk)
                    telemetry.elapsed_seconds += float(constraint_elapsed)
                    telemetry.api_modes.add("indexed_fused")
                    if unresolved_chunk > 0:
                        telemetry.rust_batch_call_count += 1
                    used_fused_path = True

                    yield _DistanceMatrixChunk(
                        block_key=block_key,
                        block_size=block_size,
                        start_offset=offset,
                        index_i=np.asarray(local_i, dtype=np.intp),
                        index_j=np.asarray(local_j, dtype=np.intp),
                        pair_ids=None,
                        labels=labels,
                        block_signature_indices=block_signature_indices,
                    )
                    offset += chunk_pair_count
                if not use_fused_block_api:
                    # Fused path disabled after optional-Rust failure; continue with non-fused chunks.
                    yield from self._yield_non_fused_chunks(
                        block_key=block_key,
                        signatures=signatures,
                        dataset=dataset,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        constraint_backend=constraint_backend,
                        telemetry=telemetry,
                        pair_chunk_size=pair_chunk_size,
                        start_offset=offset,
                    )
            else:
                yield from self._yield_non_fused_chunks(
                    block_key=block_key,
                    signatures=signatures,
                    dataset=dataset,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    constraint_backend=constraint_backend,
                    telemetry=telemetry,
                    pair_chunk_size=pair_chunk_size,
                )

        logger.info(
            "Telemetry: constraint_batch stage=distance_matrix total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f path=%s run_id=%s",
            telemetry.total_pairs,
            telemetry.partial_supervision_hits,
            telemetry.unresolved_pairs,
            telemetry.rust_batch_call_count,
            telemetry.api_mode_summary,
            telemetry.elapsed_seconds,
            "chunked_rust_fused" if used_fused_path else "chunked_rust",
            runtime_context.run_id,
        )

    def _predict_distance_matrix_chunk(
        self,
        chunk: _DistanceMatrixChunk,
        dataset: ANDData,
        runtime_context: RuntimeContext,
        batch_label: int | str,
        total_ram_bytes: int | None = None,
    ) -> tuple[np.ndarray, float]:
        use_cache = bool(self.use_cache)
        if chunk.block_signature_indices is not None and chunk.pair_ids is None:
            if use_cache:
                raise RuntimeError("Fused Rust chunk path does not support use_cache=True")
            try:
                rust_featurizer = _get_rust_featurizer(
                    dataset,
                    runtime_context=runtime_context,
                )
                selected_indices = _selected_feature_indices(self.featurizer_info)
                batch_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                    chunk.block_signature_indices,
                    start_offset=int(chunk.start_offset),
                    max_pairs=int(len(chunk.labels)),
                    selected_indices=selected_indices,
                    num_threads=self.n_jobs,
                    nan_value=np.nan,
                    featurizer=rust_featurizer,
                )
                batch_labels = np.asarray(chunk.labels, dtype=np.float64)
                batch_nameless_features: np.ndarray | None = None
                if self.nameless_classifier is not None and self.nameless_featurizer_info is not None:
                    nameless_selected_indices = _selected_feature_indices(self.nameless_featurizer_info)
                    batch_nameless_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                        chunk.block_signature_indices,
                        start_offset=int(chunk.start_offset),
                        max_pairs=int(len(chunk.labels)),
                        selected_indices=nameless_selected_indices,
                        num_threads=self.n_jobs,
                        nan_value=np.nan,
                        featurizer=rust_featurizer,
                    )
            except Exception as exc:
                if stage_uses_rust(runtime_context):
                    raise RuntimeError(
                        "Rust fused block featurization failed in strict rust backend "
                        f"(block={chunk.block_key} start_offset={chunk.start_offset} pairs={len(chunk.labels)} "
                        f"run_id={runtime_context.run_id} error={exc})"
                    ) from exc
                raise
            expected_rows = int(len(chunk.labels))
        else:
            signature_pairs = chunk.signature_pairs()
            batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
                signature_pairs,
                dataset,
                self.featurizer_info,
                self.n_jobs,
                use_cache=use_cache,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=self.nameless_featurizer_info,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            expected_rows = int(len(signature_pairs))
        batch_predictions, batch_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            batch_features,
            batch_labels,
            batch_nameless_features,
            batch_label,
            num_threads=self.n_jobs,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        if int(batch_predictions.shape[0]) != expected_rows:
            raise RuntimeError(
                "Distance-matrix chunk prediction size mismatch: "
                f"expected={expected_rows} got={batch_predictions.shape[0]}"
            )
        return np.asarray(batch_predictions, dtype=np.float64), float(batch_seconds)

    def _iter_rust_predicted_distance_matrix_chunks(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        *,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ):
        chunk_count = 0
        helper_output = self._distance_matrix_chunk_helper_rust(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
        )
        for chunk in helper_output:
            batch_predictions, batch_seconds = self._predict_distance_matrix_chunk(
                chunk,
                dataset,
                runtime_context,
                batch_label=f"chunk_{chunk_count}",
                total_ram_bytes=total_ram_bytes,
            )
            expected = int(len(chunk.labels))
            if int(batch_predictions.shape[0]) != expected:
                raise RuntimeError(
                    "Distance-matrix batch prediction count mismatch: "
                    f"expected={expected} got={batch_predictions.shape[0]}"
                )
            yield _PredictedDistanceMatrixChunk(
                chunk=chunk,
                predictions=np.asarray(batch_predictions, dtype=np.float64),
                batch_seconds=float(batch_seconds),
            )
            chunk_count += 1
        logger.info(
            "Telemetry: distance_matrix_chunking backend=rust chunks=%d run_id=%s",
            chunk_count,
            runtime_context.run_id,
        )

    def _iter_python_predicted_distance_matrix_batches(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        *,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        num_pairs: int,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ):
        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
        )
        batch_num = 0
        effective_pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
        num_batches = math.ceil(num_pairs / effective_pair_chunk_size) if num_pairs > 0 else 0
        while True:
            logger.info(f"Featurizing batch {batch_num}/{num_batches}")
            count = 0
            pairs: list[tuple[str, str, float]] = []
            indices: list[tuple[int, int]] = []
            blocks: list[str] = []
            for item in helper_output:
                pairs.append(item[0])
                indices.append(item[1])
                blocks.append(item[2])
                count += 1
                if count == effective_pair_chunk_size:
                    break

            if len(pairs) == 0:
                break

            batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
                pairs,
                dataset,
                self.featurizer_info,
                self.n_jobs,
                use_cache=self.use_cache,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=self.nameless_featurizer_info,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            batch_predictions, batch_seconds = _predict_and_combine(
                self.classifier,
                self.nameless_classifier,
                batch_features,
                batch_labels,
                batch_nameless_features,
                batch_num,
                num_threads=self.n_jobs,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            yield _PredictedDistanceMatrixBatch(
                batch_num=int(batch_num),
                blocks=blocks,
                indices=indices,
                predictions=np.asarray(batch_predictions, dtype=np.float64),
                batch_seconds=float(batch_seconds),
            )

            if count < effective_pair_chunk_size:
                break
            batch_num += 1

    def _featurize_predict_write_batches(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        *,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        num_pairs: int,
        write_prediction: Callable[[str, tuple[int, int], float], None],
        on_block_start: Callable[[str], None] | None = None,
        post_block_callback: Callable[[str], None] | None = None,
        disable_tqdm: bool = True,
        tqdm_desc: str = "Writing matrices",
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ) -> float:
        model_predict_seconds = 0.0
        prev_block_key = ""
        for batch in self._iter_python_predicted_distance_matrix_batches(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            num_pairs=num_pairs,
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
        ):
            model_predict_seconds += batch.batch_seconds
            batch_iter = tqdm(
                enumerate(batch.predictions),
                total=len(batch.predictions),
                desc=tqdm_desc,
                disable=disable_tqdm,
            )
            for within_batch_index, prediction in batch_iter:
                block_key = batch.blocks[within_batch_index]
                if block_key != prev_block_key:
                    if prev_block_key != "" and post_block_callback is not None:
                        post_block_callback(prev_block_key)
                    if on_block_start is not None:
                        on_block_start(block_key)
                write_prediction(block_key, batch.indices[within_batch_index], float(prediction))
                prev_block_key = block_key

        if prev_block_key != "" and post_block_callback is not None:
            post_block_callback(prev_block_key)
        return float(model_predict_seconds)

    def make_distance_matrices(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        disable_tqdm: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
        total_ram_bytes: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Creates the distance matrices for the input blocks.
        Note: This function is much more complicated than it needs to be in an
        effort to reduce its memory footprint

        Parameters
        ----------
        block_dict: Dict
            the block dictionary to make distances for
        dataset: ANDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        disable_tqdm: bool
            whether to turn off the tqdm progress bars in this function
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows

        Returns
        -------
        Dict: the distance matrix dictionary, keyed by block key
        """
        require_dataset_name_counts_binding_for_clusterer(
            self,
            dataset,
            context="Clusterer.make_distance_matrices",
        )
        runtime_context = dataset.runtime_context
        use_rust_blockwise = dataset_stage_uses_rust(runtime_context, dataset)
        if use_rust_blockwise:
            arrow_paths = getattr(dataset, "arrow_paths", None)
            if arrow_paths is not None:
                _require_arrow_name_counts_index_for_clusterer(
                    self,
                    arrow_paths,
                    context="Clusterer.make_distance_matrices Arrow featurizer",
                )
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)
        if partial_supervision is None:
            partial_supervision = {}
        logger.info(f"Making {len(block_dict)} distance matrices")
        logger.info("Initializing pairwise_probas")
        # initialize pairwise_probas with correctly size arrays
        pairwise_probas = {}
        num_pairs = 0
        fastcluster_dtype = np.float64 if use_rust_blockwise else np.float16
        for block_key, signatures in block_dict.items():
            block_size = len(signatures)
            num_pairs += int(block_size * (block_size - 1) / 2)
            _guard_predict_block_matrix_allocation(
                block_key=block_key,
                block_size=int(block_size),
                uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                fastcluster_fused_dtype=fastcluster_dtype,
                total_ram_bytes=total_ram_bytes,
            )
            if isinstance(self.cluster_model, FastCluster):
                # flattened pdist style
                pairwise_proba = np.zeros(int(block_size * (block_size - 1) / 2), dtype=fastcluster_dtype)
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            pairwise_probas[block_key] = pairwise_proba

        logger.info(f"Pairwise probas initialized with {num_pairs} elements, starting making all pairs")

        model_predict_seconds = 0.0
        if use_rust_blockwise:
            for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
            ):
                chunk = prediction_chunk.chunk
                batch_predictions = prediction_chunk.predictions
                model_predict_seconds += prediction_chunk.batch_seconds
                expected = int(len(chunk.labels))
                pairwise_proba = pairwise_probas[chunk.block_key]
                if isinstance(self.cluster_model, FastCluster):
                    start = int(chunk.start_offset)
                    end = start + expected
                    pairwise_proba[start:end] = np.asarray(batch_predictions, dtype=np.float64)
                else:
                    pairwise_proba[chunk.index_i, chunk.index_j] = np.asarray(
                        batch_predictions,
                        dtype=pairwise_proba.dtype,
                    )
        else:
            fastcluster_write_indices: dict[str, int] = defaultdict(int)

            def _write_prediction(
                block_key: str,
                index_pair: tuple[int, int],
                prediction: float,
            ) -> None:
                pairwise_proba = pairwise_probas[block_key]
                if isinstance(self.cluster_model, FastCluster):
                    write_index = fastcluster_write_indices[block_key]
                    if write_index >= len(pairwise_proba):
                        raise RuntimeError(
                            "FastCluster pairwise probability write overflow: "
                            f"block={block_key} index={write_index} capacity={len(pairwise_proba)}"
                        )
                    pairwise_proba[write_index] = prediction
                    fastcluster_write_indices[block_key] = write_index + 1
                else:
                    i, j = index_pair
                    pairwise_proba[i, j] = prediction

            model_predict_seconds += self._featurize_predict_write_batches(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                num_pairs=num_pairs,
                write_prediction=_write_prediction,
                disable_tqdm=disable_tqdm,
                tqdm_desc="Writing matrices",
            )

            if isinstance(self.cluster_model, FastCluster):
                for block_key, pairwise_proba in pairwise_probas.items():
                    expected_pairs = int(len(pairwise_proba))
                    observed_pairs = int(fastcluster_write_indices.get(block_key, 0))
                    if observed_pairs != expected_pairs:
                        raise RuntimeError(
                            "FastCluster pairwise probability fill mismatch: "
                            f"block={block_key} expected_pairs={expected_pairs} observed_pairs={observed_pairs}"
                        )

        if not isinstance(self.cluster_model, FastCluster):
            for pairwise_proba in pairwise_probas.values():
                pairwise_proba += pairwise_proba.T
                np.fill_diagonal(pairwise_proba, 0)

        logger.info(
            "Telemetry stage: stage=model_predict_total seconds=%.3f blocks=%d",
            model_predict_seconds,
            len(block_dict),
        )
        logger.info(f"{len(block_dict)} distance matrices made")
        return pairwise_probas

    def make_distance_matrices_from_rust_featurizer(
        self,
        block_dict: dict[str, list[str]],
        rust_featurizer: object,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Build full-predict distance matrices from an existing Rust featurizer.

        This is the ANDData-free full-predict bridge for Arrow/native Rust
        inputs. It reuses the same Rust block upper-triangle feature and
        constraint APIs as `make_distance_matrices`, but takes the already-built
        featurizer directly instead of looking it up through an `ANDData`.
        """

        require_rust_featurizer_name_counts_binding_for_clusterer(
            self,
            rust_featurizer,
            context="Clusterer.make_distance_matrices_from_rust_featurizer",
        )
        return self._make_distance_matrices_from_verified_rust_featurizer(
            block_dict,
            rust_featurizer,
            partial_supervision=partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
        )

    def _make_distance_matrices_from_verified_rust_featurizer(
        self,
        block_dict: dict[str, list[str]],
        rust_featurizer: object,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Build matrices after the request boundary verified the featurizer binding."""

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_rust_featurizer", backend="rust")
        if partial_supervision is None:
            partial_supervision = {}
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)

        signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
        selected_indices = _selected_feature_indices(self.featurizer_info)
        nameless_selected_indices = (
            _selected_feature_indices(self.nameless_featurizer_info)
            if self.nameless_classifier is not None and self.nameless_featurizer_info is not None
            else None
        )
        total_pairs = sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())
        selected_count = _count_selected_features(self.featurizer_info)
        nameless_count = (
            _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
        )
        batch_chunk_plan = _compute_predict_batch_chunk_plan(
            self.featurizer_info.number_of_features,
            selected_feature_count=selected_count,
            nameless_feature_count=nameless_count,
            total_pairs=total_pairs,
            total_ram_bytes=total_ram_bytes,
        )
        resolved_pair_chunk_size = max(
            1,
            min(
                int(self.batch_size),
                int(batch_chunk_plan.chunk_pairs) if batch_chunk_plan is not None else int(self.batch_size),
            ),
        )
        if pair_chunk_size is not None:
            resolved_pair_chunk_size = min(resolved_pair_chunk_size, max(1, int(pair_chunk_size)))

        uses_fastcluster = isinstance(self.cluster_model, FastCluster)
        fastcluster_dtype = np.float64 if uses_fastcluster else np.float16
        pairwise_probas: dict[str, np.ndarray] = {}
        model_predict_seconds = 0.0
        constraint_seconds = 0.0
        upper_triangle_index_seconds = 0.0
        label_build_seconds = 0.0
        feature_matrix_seconds = 0.0
        nameless_feature_matrix_seconds = 0.0
        matrix_write_seconds = 0.0
        chunk_count = 0
        total_start = time.perf_counter()
        for block_key, signatures in block_dict.items():
            block_size = int(len(signatures))
            block_pair_count = int(block_size * (block_size - 1) // 2)
            _guard_predict_block_matrix_allocation(
                block_key=block_key,
                block_size=block_size,
                uses_fastcluster=uses_fastcluster,
                fastcluster_fused_dtype=fastcluster_dtype,
                total_ram_bytes=total_ram_bytes,
            )
            if uses_fastcluster:
                pairwise_proba = np.zeros(block_pair_count, dtype=fastcluster_dtype)
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            pairwise_probas[block_key] = pairwise_proba
            if block_size <= 1:
                continue

            block_signature_indices = [int(signature_index_by_id[str(signature_id)]) for signature_id in signatures]
            direct_overrides, reverse_overrides = _build_partial_supervision_offset_maps_for_block(
                signatures,
                partial_supervision,
            )
            offset = 0
            while offset < block_pair_count:
                chunk_pair_count = int(min(resolved_pair_chunk_size, block_pair_count - offset))
                chunk_count += 1
                local_i_array: np.ndarray | None = None
                local_j_array: np.ndarray | None = None
                constraint_values: Sequence[Any] | None = None
                if self.use_default_constraints_as_supervision:
                    stage_start = time.perf_counter()
                    local_i, local_j, constraint_values = get_constraints_block_upper_triangle_indexed_rust(
                        block_signature_indices,
                        start_offset=offset,
                        max_pairs=chunk_pair_count,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        num_threads=self.n_jobs,
                        featurizer=rust_featurizer,
                        suppress_orcid=getattr(self, "suppress_orcid", False),
                    )
                    constraint_seconds += time.perf_counter() - stage_start
                    if not uses_fastcluster:
                        stage_start = time.perf_counter()
                        local_i_array = np.asarray(local_i, dtype=np.intp)
                        local_j_array = np.asarray(local_j, dtype=np.intp)
                        upper_triangle_index_seconds += time.perf_counter() - stage_start
                    label_count = int(len(constraint_values))
                    if label_count != chunk_pair_count:
                        raise RuntimeError(
                            "Rust constraint row count mismatch for vector write: "
                            f"block={block_key} start_offset={offset} "
                            f"expected={chunk_pair_count} got={label_count}"
                        )
                elif uses_fastcluster:
                    label_count = chunk_pair_count
                else:
                    stage_start = time.perf_counter()
                    local_i_array, local_j_array = _upper_triangle_indices_for_range(
                        block_size,
                        offset,
                        chunk_pair_count,
                    )
                    upper_triangle_index_seconds += time.perf_counter() - stage_start
                    label_count = int(len(local_i_array))

                stage_start = time.perf_counter()
                labels = np.full(label_count, np.nan, dtype=np.float64)
                if direct_overrides or reverse_overrides or constraint_values is not None:
                    for row_offset in range(label_count):
                        pair_offset = offset + row_offset
                        override = direct_overrides.get(pair_offset)
                        if override is None:
                            override = reverse_overrides.get(pair_offset)
                        if override is not None:
                            labels[row_offset] = float(override)
                            continue
                        if constraint_values is not None:
                            value = constraint_values[row_offset]
                            if value is not None:
                                labels[row_offset] = float(value - LARGE_INTEGER)
                label_build_seconds += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                batch_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                    block_signature_indices,
                    start_offset=offset,
                    max_pairs=chunk_pair_count,
                    selected_indices=selected_indices,
                    num_threads=self.n_jobs,
                    nan_value=np.nan,
                    featurizer=rust_featurizer,
                )
                feature_matrix_seconds += time.perf_counter() - stage_start
                batch_nameless_features: np.ndarray | None = None
                if nameless_selected_indices is not None:
                    stage_start = time.perf_counter()
                    batch_nameless_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                        block_signature_indices,
                        start_offset=offset,
                        max_pairs=chunk_pair_count,
                        selected_indices=nameless_selected_indices,
                        num_threads=self.n_jobs,
                        nan_value=np.nan,
                        featurizer=rust_featurizer,
                    )
                    nameless_feature_matrix_seconds += time.perf_counter() - stage_start
                batch_predictions, batch_seconds = _predict_and_combine(
                    self.classifier,
                    self.nameless_classifier,
                    batch_features,
                    labels,
                    batch_nameless_features,
                    f"{block_key}:{offset}",
                    num_threads=self.n_jobs,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
                model_predict_seconds += float(batch_seconds)
                stage_start = time.perf_counter()
                if uses_fastcluster:
                    end = offset + int(len(batch_predictions))
                    pairwise_proba[offset:end] = np.asarray(batch_predictions, dtype=pairwise_proba.dtype)
                else:
                    if local_i_array is None or local_j_array is None:
                        raise RuntimeError("Non-FastCluster matrix write requires upper-triangle index arrays")
                    pairwise_proba[local_i_array, local_j_array] = np.asarray(
                        batch_predictions,
                        dtype=pairwise_proba.dtype,
                    )
                matrix_write_seconds += time.perf_counter() - stage_start
                offset += chunk_pair_count

            if not uses_fastcluster:
                pairwise_proba += pairwise_proba.T
                np.fill_diagonal(pairwise_proba, 0)

        logger.info(
            "Telemetry stage: stage=model_predict_rust_featurizer_total seconds=%.3f blocks=%d",
            model_predict_seconds,
            len(block_dict),
        )
        telemetry = {
            "total_seconds": float(time.perf_counter() - total_start),
            "constraint_seconds": float(constraint_seconds),
            "upper_triangle_index_seconds": float(upper_triangle_index_seconds),
            "label_build_seconds": float(label_build_seconds),
            "feature_matrix_seconds": float(feature_matrix_seconds),
            "nameless_feature_matrix_seconds": float(nameless_feature_matrix_seconds),
            "model_predict_seconds": float(model_predict_seconds),
            "matrix_write_seconds": float(matrix_write_seconds),
            "block_count": int(len(block_dict)),
            "pair_count": int(total_pairs),
            "chunk_count": int(chunk_count),
            "resolved_pair_chunk_size": int(resolved_pair_chunk_size),
        }
        self._last_rust_featurizer_make_dists_telemetry = telemetry
        logger.info(
            "Telemetry stage: stage=rust_featurizer_make_dists total_seconds=%.3f "
            "constraint_seconds=%.3f upper_triangle_index_seconds=%.3f "
            "label_build_seconds=%.3f feature_matrix_seconds=%.3f "
            "nameless_feature_matrix_seconds=%.3f model_predict_seconds=%.3f "
            "matrix_write_seconds=%.3f blocks=%d pairs=%d chunks=%d pair_chunk_size=%d",
            telemetry["total_seconds"],
            telemetry["constraint_seconds"],
            telemetry["upper_triangle_index_seconds"],
            telemetry["label_build_seconds"],
            telemetry["feature_matrix_seconds"],
            telemetry["nameless_feature_matrix_seconds"],
            telemetry["model_predict_seconds"],
            telemetry["matrix_write_seconds"],
            telemetry["block_count"],
            telemetry["pair_count"],
            telemetry["chunk_count"],
            telemetry["resolved_pair_chunk_size"],
        )
        return pairwise_probas

    def predict_from_rust_featurizer(
        self,
        block_dict: dict[str, list[str]],
        rust_featurizer: object,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        cluster_seeds_require: Mapping[str, int | str] | None = None,
        cluster_seeds_disallow: set[tuple[str, str]] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """Predict full blocks from an already-built Rust featurizer."""

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_rust_featurizer", backend="rust")
        if partial_supervision is None:
            partial_supervision = {}
        built_dists = dists is None
        if built_dists:
            require_rust_featurizer_name_counts_binding_for_clusterer(
                self,
                rust_featurizer,
                context="Clusterer.predict_from_rust_featurizer",
            )
        resolved_cluster_seeds_require: dict[str, int | str]
        if cluster_seeds_require is None:
            resolved_cluster_seeds_require = _cluster_seeds_require_from_rust_featurizer(rust_featurizer)
        else:
            resolved_cluster_seeds_require = {
                str(signature_id): component_id for signature_id, component_id in cluster_seeds_require.items()
            }
        resolved_cluster_seeds_disallow = set(normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow or ()))
        explicit_cluster_seeds_require = (
            None
            if cluster_seeds_require is None
            else {str(signature_id): component_id for signature_id, component_id in cluster_seeds_require.items()}
        )
        if dists is not None and resolved_cluster_seeds_disallow:
            raise ValueError(
                "cluster_seeds_disallow cannot be used with precomputed dists because disallow pairs "
                "would not be injected into the distance matrix"
            )
        if dists is not None and resolved_cluster_seeds_require and not incremental_dont_use_cluster_seeds:
            raise ValueError(
                "cluster_seeds_require cannot be used with precomputed dists because require pairs "
                "would not be injected into the distance matrix"
            )
        effective_cluster_seeds_require = {} if incremental_dont_use_cluster_seeds else resolved_cluster_seeds_require
        proxy_dataset = _RustFeaturizerPredictDataset(
            cluster_seeds_require=effective_cluster_seeds_require,
            cluster_seeds_disallow=resolved_cluster_seeds_disallow,
            signatures=_signature_rule_metadata_from_rust_featurizer(rust_featurizer),
        )
        if built_dists:
            pred_clusters: defaultdict[str, list[str]] = defaultdict(list)
            all_disallow_signature_ids: set[str] = set()
            if self.use_default_constraints_as_supervision:
                for sig_id_a, sig_id_b in proxy_dataset.cluster_seeds_disallow:
                    all_disallow_signature_ids.add(sig_id_a)
                    all_disallow_signature_ids.add(sig_id_b)
            effective_cluster_model_params = cluster_model_params
            if isinstance(self.cluster_model, FastCluster):
                fastcluster_params: dict[str, Any] = dict(cluster_model_params or {})
                fastcluster_params.setdefault("preserve_input", False)
                effective_cluster_model_params = fastcluster_params
            make_dists_seconds = 0.0
            cluster_seconds = 0.0
            make_dists_telemetry: dict[str, int | float | str] = {}
            for block_key, signatures in block_dict.items():
                make_block_start = time.perf_counter()
                block_partial_supervision = _partial_supervision_with_cluster_seed_overrides(
                    signatures,
                    partial_supervision,
                    cluster_seeds_require=explicit_cluster_seeds_require,
                    cluster_seeds_disallow=resolved_cluster_seeds_disallow,
                    dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                )
                make_dists_incremental_dont_use_cluster_seeds = (
                    incremental_dont_use_cluster_seeds or explicit_cluster_seeds_require is not None
                )
                block_dists = self._make_distance_matrices_from_verified_rust_featurizer(
                    {block_key: signatures},
                    rust_featurizer,
                    partial_supervision=block_partial_supervision,
                    incremental_dont_use_cluster_seeds=make_dists_incremental_dont_use_cluster_seeds,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
                make_dists_seconds += time.perf_counter() - make_block_start
                block_make_dists_telemetry = dict(getattr(self, "_last_rust_featurizer_make_dists_telemetry", {}) or {})
                for key, value in block_make_dists_telemetry.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        make_dists_telemetry[key] = float(make_dists_telemetry.get(key, 0.0)) + float(value)
                    elif key not in make_dists_telemetry:
                        make_dists_telemetry[key] = value
                cluster_block_start = time.perf_counter()
                labels = self._cluster_one_block_with_logging(
                    signatures,
                    block_dists[block_key],
                    effective_cluster_model_params,
                    cast(Any, proxy_dataset),
                    all_disallow_signature_ids,
                    block_key=block_key,
                )
                cluster_seconds += time.perf_counter() - cluster_block_start
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
        self._last_rust_featurizer_predict_telemetry = telemetry
        logger.info(
            "Telemetry stage: stage=rust_featurizer_predict make_dists_seconds=%.3f "
            "cluster_from_dists_seconds=%.3f blocks=%d pairs=%d",
            telemetry["make_dists_seconds"],
            telemetry["cluster_from_dists_seconds"],
            telemetry["block_count"],
            telemetry["pair_count"],
        )
        return dict(pred_clusters), out_dists

    def predict_from_arrow_paths(
        self,
        block_dict: dict[str, list[str]],
        arrow_paths: Mapping[str, Any],
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        load_name_counts: bool | None = None,
        name_tuples: set[tuple[str, str]] | str | None = "filtered",
        cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """Predict full blocks directly from Arrow IPC inputs through Rust."""

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_from_arrow_paths", backend="rust")
        elif not stage_uses_rust(runtime_context):
            raise ValueError("Clusterer.predict_from_arrow_paths requires a Rust runtime context")
        resolved_load_name_counts = resolve_load_name_counts_policy(
            self,
            load_name_counts,
            context="Clusterer.predict_from_arrow_paths",
        )
        arrow_path_payload = validate_arrow_prediction_artifacts(
            arrow_paths,
            require_specter=clusterer_uses_embedding_features(self),
            require_name_counts_index=resolved_load_name_counts,
            expected_normalization_version=_resolve_clusterer_normalization_version(self),
            context="Clusterer.predict_from_arrow_paths",
            producer_hint=(
                "provide signatures, papers, paper_authors, model-required specter, and model-required "
                "name_counts_index plus raw-planner batch indexes from scripts/convert_to_arrow.py or the "
                "published s2and-release-arrow bundle"
            ),
        )
        _require_arrow_name_counts_index_for_clusterer(self, arrow_path_payload, context="Arrow prediction")
        signature_ids = list(
            dict.fromkeys(str(signature_id) for signatures in block_dict.values() for signature_id in signatures)
        )
        arrow_cluster_seed_disallows = _cluster_seed_disallows_from_arrow_paths(arrow_path_payload)
        cluster_seed_disallows = set(arrow_cluster_seed_disallows)
        if cluster_seeds_disallow is not None:
            cluster_seed_disallows.update(normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow))
        if dists is not None and cluster_seed_disallows:
            raise ValueError(
                "cluster_seeds_disallow cannot be used with precomputed dists because disallow pairs "
                "would not be injected into the distance matrix"
            )
        effective_partial_supervision = _partial_supervision_with_cluster_seed_disallows(
            signature_ids,
            self,
            partial_supervision or {},
            arrow_path_payload,
            cluster_seed_disallows=arrow_cluster_seed_disallows,
        )
        featurizer_start = time.perf_counter()
        rust_featurizer = build_rust_featurizer_from_arrow_paths(
            arrow_path_payload,
            expected_normalization_version=_resolve_clusterer_normalization_version(self),
            signature_ids=signature_ids,
            name_tuples=name_tuples,
            load_name_counts=resolved_load_name_counts,
            preprocess=True,
            num_threads=self.n_jobs,
        )
        arrow_featurizer_seconds = time.perf_counter() - featurizer_start
        predict_start = time.perf_counter()
        result = self.predict_from_rust_featurizer(
            block_dict,
            rust_featurizer,
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=effective_partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            cluster_seeds_disallow=cluster_seed_disallows,
        )
        predict_seconds = time.perf_counter() - predict_start
        nested_telemetry = dict(getattr(self, "_last_rust_featurizer_predict_telemetry", {}) or {})
        telemetry = {
            "arrow_featurizer_seconds": float(arrow_featurizer_seconds),
            "rust_featurizer_predict_seconds": float(predict_seconds),
            "total_seconds": float(arrow_featurizer_seconds + predict_seconds),
            "signature_count": int(len(signature_ids)),
            "block_count": int(len(block_dict)),
            "pair_count": int(sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())),
            **{f"rust_{key}": value for key, value in nested_telemetry.items()},
        }
        self._last_arrow_predict_telemetry = telemetry
        logger.info(
            "Telemetry stage: stage=arrow_predict_total total_seconds=%.3f "
            "arrow_featurizer_seconds=%.3f rust_featurizer_predict_seconds=%.3f "
            "signatures=%d blocks=%d pairs=%d",
            telemetry["total_seconds"],
            telemetry["arrow_featurizer_seconds"],
            telemetry["rust_featurizer_predict_seconds"],
            telemetry["signature_count"],
            telemetry["block_count"],
            telemetry["pair_count"],
        )
        return result

    def fit(
        self,
        datasets: ANDData | list[ANDData],
        val_dists_precomputed: dict[str, dict[str, np.ndarray]] | None = None,
        metric_for_hyperopt: str = "b3",
    ) -> Clusterer:
        """
        Fits the clusterer

        Parameters
        ----------
        datasets: List[ANDData]
            the list of datasets to use for validations
        val_dists_precomputed: Dict
            precomputed distance matrices
        metric_for_hyperopt: string
            the metric to use for hyperparamter optimization

        Returns
        -------
        Clusterer: a fit clusterer, also sets the best params
        """
        assert metric_for_hyperopt in {"b3", "ratio"}
        logger.info("Fitting clusterer")
        if isinstance(datasets, ANDData):
            datasets = [datasets]
        if len(datasets) == 0:
            raise ValueError("Clusterer.fit requires at least one dataset")

        dataset_normalization_versions = {getattr(dataset, "normalization_version", None) for dataset in datasets}
        if len(dataset_normalization_versions) != 1:
            raise ValueError(
                "Clusterer.fit requires one dataset normalization version; "
                f"observed={sorted(repr(value) for value in dataset_normalization_versions)}"
            )
        training_normalization_version = next(iter(dataset_normalization_versions))
        if training_normalization_version != NORMALIZATION_VERSION:
            raise ValueError(
                f"Clusterer.fit requires dataset normalization_version={NORMALIZATION_VERSION!r}; "
                f"observed={training_normalization_version!r}"
            )
        contract = getattr(self, "feature_contract", None)
        if not isinstance(contract, dict):
            contract = {}
        contract["normalization_version"] = training_normalization_version
        if clusterer_uses_name_count_features(self):
            count_provenances = [
                validated_name_counts_provenance(
                    getattr(dataset, "name_counts_provenance", None),
                    context=f"Clusterer.fit dataset={getattr(dataset, 'name', '<unnamed>')}",
                )
                for dataset in datasets
            ]
            count_versions = {value.get("normalization_version") for value in count_provenances}
            generation_ids = {value.get("generation_id") for value in count_provenances}
            pickle_digests = {value.get("pickle_sha256") for value in count_provenances}
            snapshot_ids = {value.get("source_snapshot_id") for value in count_provenances}
            selected_row_digests = {value.get("selected_rows_sha256") for value in count_provenances}
            if (
                count_versions != {training_normalization_version}
                or len(generation_ids) != 1
                or len(pickle_digests) != 1
                or len(snapshot_ids) != 1
                or len(selected_row_digests) != 1
            ):
                raise ValueError(
                    "Clusterer.fit requires one verified name-count generation matching dataset normalization; "
                    f"versions={count_versions!r} generations={generation_ids!r} digests={pickle_digests!r} "
                    f"snapshots={snapshot_ids!r} row_digests={selected_row_digests!r}"
                )
            contract["name_counts_generation_id"] = next(iter(generation_ids))
            contract["name_counts_pickle_sha256"] = next(iter(pickle_digests))
            contract["name_counts_source_snapshot_id"] = next(iter(snapshot_ids))
            contract["name_counts_selected_rows_sha256"] = next(iter(selected_row_digests))
        self.feature_contract = contract

        val_block_dict_list = []
        val_cluster_to_signatures_list = []
        val_dists_list = []
        val_datasets_list: list[ANDData] = []
        weights: list[float] = []
        for dataset in datasets:
            # blocks
            train_block_dict, val_block_dict, _ = dataset.split_cluster_signatures()
            # incremental setting uses all the signatures in train and val
            # block-wise split uses only validation set for building the clustering model
            if dataset.unit_of_data_split == "time" or dataset.unit_of_data_split == "signatures":
                for block_key, signatures in train_block_dict.items():
                    if block_key in val_block_dict:
                        val_block_dict[block_key].extend(signatures)

            # we don't need val blocks with only a single element
            val_block_dict = self.filter_blocks(val_block_dict, self.val_blocks_size)
            val_block_dict_list.append(val_block_dict)

            # block ground truth labels: cluster_to_signatures
            val_cluster_to_signatures = dataset.construct_cluster_to_signatures(val_block_dict)
            val_cluster_to_signatures_list.append(val_cluster_to_signatures)

            # distance matrix
            if val_dists_precomputed is None:
                val_dists = self.make_distance_matrices(val_block_dict, dataset)
            else:
                val_dists = val_dists_precomputed[dataset.name]
            val_dists_list.append(val_dists)
            val_datasets_list.append(dataset)

            # weights for weighted F1 average: total # of signatures in dataset
            weights.append(np.sum([len(i) for i in val_block_dict.values()]))

        def obj(params):
            self.set_params(params)
            f1s = []
            ratios = []
            for val_dataset, val_block_dict, val_cluster_to_signatures, val_dists in zip(
                val_datasets_list, val_block_dict_list, val_cluster_to_signatures_list, val_dists_list, strict=True
            ):
                pred_clusters, _ = self.predict(
                    val_block_dict,
                    dataset=val_dataset,
                    dists=val_dists,
                )
                (
                    _,
                    _,
                    f1,
                    _,
                    pred_bigger_ratios,
                    true_bigger_ratios,
                ) = b3_precision_recall_fscore(val_cluster_to_signatures, pred_clusters)
                ratios.append(np.mean(pred_bigger_ratios + true_bigger_ratios))
                f1s.append(f1)
            if metric_for_hyperopt == "ratio":
                return np.average(ratios, weights=weights)
            elif metric_for_hyperopt == "b3":
                # minimize means we need to negate
                return -np.average(f1s, weights=weights)

        self.hyperopt_trials_store = Trials()
        _ = fmin(
            fn=obj,
            space=self.search_space,
            algo=partial(tpe.suggest, n_startup_jobs=5),
            max_evals=self.n_iter,
            trials=self.hyperopt_trials_store,
            rstate=np.random.default_rng(self.random_state),
        )
        # hyperopt has some problems with hp.choice so we need to do this:
        assert isinstance(self.hyperopt_trials_store, Trials)
        best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
        self.best_params = {k: intify(v) for k, v in best_params.items()}
        self.set_params(self.best_params)

        logger.info("Clusterer fit")
        return self

    def set_params(self, params: dict[str, Any] | None, clone_flag: bool = False):
        """
        Sets params on the cluster model

        Parameters
        ----------
        params: Dict
            the params to set
        clone_flag: bool
            whether to return a clone of the cluster model
        """
        if params is None:
            params = {}
        else:
            params = {k: intify(v) for k, v in params.items()}
        if clone_flag:
            cluster_model = clone(self.cluster_model)
            cluster_model.set_params(**params)
            return cluster_model
        else:
            self.cluster_model.set_params(**params)

    def _build_subblocked_block_dict(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        *,
        batching_threshold: int,
        specter_cluster_fn: Callable[..., dict[str, list[str]]] | None = None,
    ) -> dict[str, list[str]]:
        block_dict_subblocked: dict[str, list[str]] = {}
        for block_key in sorted(block_dict):
            block_signatures = block_dict[block_key]
            if len(block_signatures) > batching_threshold:
                kwargs: dict[str, Any] = {"maximum_size": batching_threshold}
                if specter_cluster_fn is not None:
                    kwargs["specter_cluster_fn"] = specter_cluster_fn
                subblocks = make_subblocks(block_signatures, dataset, **kwargs)
                for subblock_key in sorted(subblocks):
                    subblock_signatures = subblocks[subblock_key]
                    block_dict_subblocked[f"{block_key}|subblock={subblock_key}"] = subblock_signatures
                    assert len(subblock_signatures) <= batching_threshold, "Subblock is too big for some reason!"
            else:
                block_dict_subblocked[block_key] = block_signatures
        return block_dict_subblocked

    def _subblocking_graph_config(self) -> GraphSubblockingConfig:
        return _resolve_graph_subblocking_config(getattr(self, "subblocking_graph_config", None))

    def _subblocking_specter_cluster_fn(
        self,
        arrow_paths: Mapping[str, Any] | None,
        signature_ids: Sequence[str],
    ) -> Callable[..., dict[str, list[str]]] | None:
        candidate_signature_count = int(len(tuple(dict.fromkeys(str(value) for value in signature_ids))))
        if arrow_paths is not None:
            fallback = make_arrow_graph_subblocking_cluster_fn(
                arrow_paths,
                signature_ids,
                config=self._subblocking_graph_config(),
                random_seed=int(getattr(self, "random_state", 0) or 0),
            )
            source = "arrow"
        else:
            fallback = make_dataset_graph_subblocking_cluster_fn(config=self._subblocking_graph_config())
            source = "anddata"
        adapter = _GraphSubblockingFallbackWithLegacyFallback(
            fallback,
            source=source,
            candidate_signature_count=candidate_signature_count,
        )
        self._last_graph_subblocking_telemetry = adapter.telemetry
        self._last_arrow_graph_subblocking_telemetry = self._last_graph_subblocking_telemetry
        return adapter

    def _partition_subblocked_first_name_groups(
        self,
        block_dict_subblocked: dict[str, list[str]],
        dataset: ANDData,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], bool]:
        nonempty_subblocks = {
            block_key: block_signatures
            for block_key, block_signatures in block_dict_subblocked.items()
            if len(block_signatures) > 0
        }
        single_letter = {
            block_key: block_signatures
            for block_key, block_signatures in nonempty_subblocks.items()
            if len(_signature_first_for_rules(dataset.signatures[block_signatures[0]])) <= 1
        }
        multiple_letter = {
            block_key: block_signatures
            for block_key, block_signatures in nonempty_subblocks.items()
            if block_key not in single_letter
        }
        if len(multiple_letter) == 0:
            return single_letter, {}, True
        return multiple_letter, single_letter, False

    def _predict_subblocked_multiple_letter_groups(
        self,
        block_dict_multiple_letter: dict[str, list[str]],
        *,
        alert_flag: bool,
        dataset: ANDData,
        cluster_model_params: dict[str, Any] | None,
        partial_supervision: dict[tuple[str, str], int | float],
        use_s2_clusters: bool,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None,
        rust_featurizer: object | None = None,
        cluster_seeds_disallow: set[tuple[str, str]] | None = None,
    ) -> dict[str, list[str]]:
        pred_clusters: dict[str, list[str]] = {}
        if len(block_dict_multiple_letter) == 0:
            return pred_clusters

        if alert_flag:
            logger.info("Note! There are no subblocks with multiple letter first names")
            logger.info("Running predict on subblocks with single letter first names")
        else:
            logger.info("Running predict on subblocks with multiple letter first names")

        predict_times: dict[str, float] = {}
        for block_key in sorted(block_dict_multiple_letter):
            block_signatures = block_dict_multiple_letter[block_key]
            logger.info(f"Working on subblock {block_key}")
            start = time.time()
            if rust_featurizer is None:
                pred_clusters_intermediate, _ = self.predict_helper(
                    {block_key: block_signatures},
                    dataset,
                    dists=None,
                    cluster_model_params=cluster_model_params,
                    partial_supervision=partial_supervision,
                    use_s2_clusters=use_s2_clusters,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
            else:
                pred_clusters_intermediate, _ = self.predict_from_rust_featurizer(
                    {block_key: block_signatures},
                    rust_featurizer,
                    dists=None,
                    cluster_model_params=cluster_model_params,
                    partial_supervision=partial_supervision,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                    cluster_seeds_disallow=cluster_seeds_disallow,
                )
            end = time.time()
            predict_times[block_key] = end - start
            pred_clusters.update(pred_clusters_intermediate)
        logger.info(f"Finished, here's how long each took: {predict_times}")
        return pred_clusters

    def _predict_subblocked_single_letter_incremental_groups(
        self,
        block_dict_single_letter: dict[str, list[str]],
        *,
        pred_clusters: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> dict[str, list[str]]:
        if len(block_dict_single_letter) == 0:
            return pred_clusters

        logger.info("Running predict incremental on subblocks with single letter first names")
        cluster_seeds_require_original = copy.deepcopy(dataset.cluster_seeds_require)
        altered_cluster_signatures_was_present = hasattr(dataset, "altered_cluster_signatures")
        altered_cluster_signatures_original = getattr(dataset, "altered_cluster_signatures", None)
        dataset.cluster_seeds_require = {}
        # This is bulk subblocking's synthetic incremental pass. The original altered-profile
        # list and Arrow seed path are keyed to production claimed-profile seeds, not to
        # these temporary clusters.
        dataset.altered_cluster_signatures = []
        pred_clusters_intermediate: dict[str, list[str]] = pred_clusters
        try:
            for cluster_id, signatures in pred_clusters_intermediate.items():
                for signature in signatures:
                    dataset.cluster_seeds_require[signature] = cluster_id
            _bump_cluster_seeds_version(dataset)

            predict_times: dict[str, float] = {}
            for block_key in sorted(block_dict_single_letter.keys()):
                block_signatures = block_dict_single_letter[block_key]
                start_predict_time = time.time()
                incremental_result = self.predict_incremental(
                    block_signatures,
                    dataset,
                    prevent_new_incompatibilities=True,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
                clusters_payload = incremental_result.get("clusters")
                if not isinstance(clusters_payload, dict):
                    raise RuntimeError(
                        "predict_incremental returned invalid clusters payload; expected dict "
                        f"got {type(clusters_payload).__name__}"
                    )
                pred_clusters_intermediate = {}
                for cluster_id, signatures in clusters_payload.items():
                    if not isinstance(signatures, list):
                        raise RuntimeError(
                            "predict_incremental returned invalid cluster member payload; expected list "
                            f"for cluster_id={cluster_id!r}, got {type(signatures).__name__}"
                        )
                    pred_clusters_intermediate[str(cluster_id)] = [str(signature) for signature in signatures]
                end_predict_time = time.time()
                predict_times[block_key] = end_predict_time - start_predict_time

                dataset.cluster_seeds_require = {}
                for cluster_id, signatures in pred_clusters_intermediate.items():
                    for signature in signatures:
                        dataset.cluster_seeds_require[signature] = cluster_id
                _bump_cluster_seeds_version(dataset)

            logger.info(f"Finished subblocked predict incremental. Here's how long each subblock took: {predict_times}")
        finally:
            dataset.cluster_seeds_require = cluster_seeds_require_original
            if altered_cluster_signatures_was_present:
                dataset.altered_cluster_signatures = altered_cluster_signatures_original
            elif hasattr(dataset, "altered_cluster_signatures"):
                delattr(dataset, "altered_cluster_signatures")
            _bump_cluster_seeds_version(dataset)
            _ensure_cluster_seed_version_tracking(dataset)
        return pred_clusters_intermediate

    def _predict_subblocked(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        *,
        cluster_model_params: dict[str, Any] | None,
        partial_supervision: dict[tuple[str, str], int | float],
        use_s2_clusters: bool,
        incremental_dont_use_cluster_seeds: bool,
        batching_threshold: int,
        runtime_context: RuntimeContext,
        dists: dict[str, np.ndarray] | None,
        total_ram_bytes: int | None,
    ) -> tuple[dict[str, list[str]], None]:
        if stage_uses_rust(runtime_context):
            raise ValueError("Classic subblocked prediction requires a Python runtime context")
        assert batching_threshold > 0, "Batching threshold must be positive"
        assert dists is None, "If batching_threshold is not None, then can't use precomputed dists"
        self._last_subblocked_altered_presplit_telemetry = {
            "bulk_altered_presplit_applied": 0,
            "bulk_altered_presplit_seconds": 0.0,
        }
        subblocking_signature_ids = list(
            dict.fromkeys(str(signature_id) for signatures in block_dict.values() for signature_id in signatures)
        )
        specter_cluster_fn = self._subblocking_specter_cluster_fn(None, subblocking_signature_ids)
        block_dict_subblocked = self._build_subblocked_block_dict(
            block_dict,
            dataset,
            batching_threshold=batching_threshold,
            specter_cluster_fn=specter_cluster_fn,
        )
        (
            block_dict_multiple_letter_first_names,
            block_dict_single_letter_first_names,
            alert_flag,
        ) = self._partition_subblocked_first_name_groups(block_dict_subblocked, dataset)
        pred_clusters = self._predict_subblocked_multiple_letter_groups(
            block_dict_multiple_letter_first_names,
            alert_flag=alert_flag,
            dataset=dataset,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            use_s2_clusters=use_s2_clusters,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            rust_featurizer=None,
            cluster_seeds_disallow=set(),
        )
        pred_clusters = self._predict_subblocked_single_letter_incremental_groups(
            block_dict_single_letter_first_names,
            pred_clusters=pred_clusters,
            dataset=dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        return dict(pred_clusters), None

    def predict(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
        batching_threshold: int | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """
        Predicts clusters

        Parameters
        ----------
        block_dict: Dict
            the block dict to predict clusters from
        dataset: ANDData
            the dataset
        dists: Dict
            (optional) precomputed distance matrices
        cluster_model_params: Dict
            params to set on the cluster model
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        use_s2_clusters: bool
            whether to "predict" using the clusters from Semantic Scholar's old system
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows
        batching_threshold: int | None
            If the number of signatures in a block is above this number, we will use subblocking on the block.
            This means that the single-letter first names will be sent through via predict_incremental.
            Defaults to None, which means no batching occurs
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for exact/non-incremental predict paths. When set, predict_helper
            uses it for pair-batch sizing and fails fast before allocating a block distance matrix that
            would exceed the usable budget.
        Note: batching_threshold is a hack to get around OOM issues. We will assume that it implies
        that we don't want to ever take up more memory than (batching_threshold ** 2)

        Returns
        -------
        Dict: the predicted clusters
        Optional[Dict]: the predicted distance matrices. This is None when
        distances are built and clustered in the fused one-block-at-a-time path.
        """

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict", backend="python")
        elif stage_uses_rust(runtime_context):
            raise ValueError(
                "Clusterer.predict operates on ANDData through Python; use predict_from_arrow_paths for Rust"
            )

        if partial_supervision is None:
            partial_supervision = {}

        if dists is None and not use_s2_clusters and batching_threshold is not None:
            require_dataset_name_counts_binding_for_clusterer(
                self,
                dataset,
                context="Clusterer.predict",
            )

        if batching_threshold is not None:
            pred_clusters, dists = self._predict_subblocked(
                block_dict,
                dataset,
                cluster_model_params=cluster_model_params,
                partial_supervision=partial_supervision,
                use_s2_clusters=use_s2_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                batching_threshold=int(batching_threshold),
                runtime_context=runtime_context,
                dists=dists,
                total_ram_bytes=total_ram_bytes,
            )

        else:
            # normal mode - everything goes through full block clustering
            logger.info("Running predict on full blocks - no subblocking")
            start = time.time()
            pred_clusters, dists = self.predict_helper(
                block_dict,
                dataset,
                dists=dists,
                cluster_model_params=cluster_model_params,
                partial_supervision=partial_supervision,
                use_s2_clusters=use_s2_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            end = time.time()
            total_predict_time = end - start
            logger.info(f"Finished predict on full blocks. Time taken: {total_predict_time}")

        return dict(pred_clusters), dists

    def _cluster_one_block(
        self,
        block_signatures: list[str],
        dist_matrix: np.ndarray | None,
        cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
    ) -> list:
        """Cluster one block from a distance matrix and return labels."""
        if len(block_signatures) == 0:
            return []
        if len(block_signatures) == 1:
            return [0]

        if dist_matrix is None:
            raise ValueError("Distance matrix is required for blocks with more than one signature.")

        cluster_model = clone(self.cluster_model)
        params = {k: intify(v) for k, v in (cluster_model_params or {}).items()}
        cluster_model.set_params(**params)
        cluster_model.fit(dist_matrix)
        labels = cluster_model.labels_
        max_label = labels.max()
        # In HDBSCAN, label -1 denotes outliers.
        # Give each outlier its own unique label starting at max_label + 1.
        negative_one_label_locations = np.where(labels == -1)[0]
        for i, loc in enumerate(negative_one_label_locations):
            labels[loc] = max_label + 1 + i
        if self.use_default_constraints_as_supervision:
            disallow_signature_ids = all_disallow_signature_ids
            inverse_id_map = defaultdict(set)
            for signature_id, label in zip(block_signatures, labels, strict=True):
                if signature_id in dataset.cluster_seeds_require and signature_id not in disallow_signature_ids:
                    inverse_id_map[dataset.cluster_seeds_require[signature_id]].add(label)
            # Clusters that should merge can still remain split after distance-based clustering.
            # This happens when required-pair zero distances are outweighed by many large distances
            # in average-linkage behavior. Post-hoc, merge label sets that overlap according to
            # cluster_seeds_require (excluding signatures that appear in disallow constraints).
            to_join_sets = [sorted(join_set) for join_set in inverse_id_map.values() if len(join_set) > 1]
            mapped_labels = {label: label for label in labels}
            labels = np.array(labels)
            for join_set in to_join_sets:
                for other_label in join_set[1:]:
                    labels[labels == mapped_labels[other_label]] = mapped_labels[join_set[0]]
                    mapped_labels[other_label] = mapped_labels[join_set[0]]
            labels = list(labels)
        return labels

    def _cluster_one_block_with_logging(
        self,
        block_signatures: list[str],
        dist_matrix: np.ndarray | None,
        cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
        *,
        block_key: str,
    ) -> list:
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
            all_disallow_signature_ids,
        )
        logger.info(
            "Finished cluster_model.fit for block %s using %s in %.3fs (clusters=%d)",
            block_key,
            cluster_model_name,
            time.perf_counter() - cluster_start,
            len(set(labels)),
        )
        return labels

    def predict_helper(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """
        Predicts clusters

        Parameters
        ----------
        block_dict: Dict
            the block dict to predict clusters from
        dataset: ANDData
            the dataset
        dists: Dict
            (optional) precomputed distance matrices
        cluster_model_params: Dict
            params to set on the cluster model
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        use_s2_clusters: bool
            whether to "predict" using the clusters from Semantic Scholar's old system
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for pair-batch sizing and exact block-matrix allocation checks.

        Returns
        -------
        Dict: the predicted clusters
        Optional[Dict]: the predicted distance matrices. This is None when
        distances are built and clustered in the fused one-block-at-a-time path.
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict", backend="python")

        if partial_supervision is None:
            partial_supervision = {}
        if dists is not None and partial_supervision:
            raise ValueError(
                "partial_supervision cannot be used with precomputed dists because override pairs "
                "would not be injected into the distance matrix"
            )
        if dists is None and not use_s2_clusters:
            require_dataset_name_counts_binding_for_clusterer(
                self,
                dataset,
                context="Clusterer.predict_helper",
            )

        pred_clusters = defaultdict(list)

        if use_s2_clusters:
            for _, signature_list in block_dict.items():
                for _signature in signature_list:
                    s2_cluster_key = dataset.signatures[_signature].author_id
                    pred_clusters[s2_cluster_key].append(_signature)

            return dict(pred_clusters), dists

        # we may need this set later for post-hoc merging
        # pre-compute disallow set for post-hoc constraint merging
        all_disallow_signature_ids: set[str] = set()
        if self.use_default_constraints_as_supervision:
            for sig_id_a, sig_id_b in dataset.cluster_seeds_disallow:
                all_disallow_signature_ids.add(sig_id_a)
                all_disallow_signature_ids.add(sig_id_b)

        effective_cluster_model_params = cluster_model_params
        fastcluster_fused_dtype = np.float16
        if isinstance(self.cluster_model, FastCluster):
            fastcluster_params: dict[str, Any] = dict(cluster_model_params or {})
            # Reused matrices should stay immutable by default; single-use fused path favors lower peak memory.
            if "preserve_input" not in fastcluster_params:
                fastcluster_params["preserve_input"] = bool(dists is not None)
            effective_cluster_model_params = fastcluster_params
            if dists is None:
                fastcluster_fused_dtype = np.float64

        if dists is not None:
            # precomputed dists (hyperopt path) — cluster from existing matrices
            for block_key in block_dict.keys():
                if block_key not in dists:
                    raise KeyError(f"Missing precomputed distance matrix for block '{block_key}'")
                labels = self._cluster_one_block_with_logging(
                    block_dict[block_key],
                    dists[block_key],
                    effective_cluster_model_params,
                    dataset,
                    all_disallow_signature_ids,
                    block_key=block_key,
                )
                for signature, label in zip(block_dict[block_key], labels, strict=True):
                    pred_clusters[block_key + "_" + str(label)].append(signature)
            return dict(pred_clusters), dists

        # fused path: build one block's matrix, cluster it, free it, repeat
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)

        prev_block_key = ""
        pairwise_proba: np.ndarray | None = None
        seen_block_keys: set[str] = set()
        num_pairs = sum(len(sigs) * (len(sigs) - 1) // 2 for sigs in block_dict.values())
        logger.info("Predict helper: total_pairs=%d", num_pairs)
        model_predict_seconds = 0.0
        selected_count = _count_selected_features(self.featurizer_info)
        nameless_count = (
            _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
        )
        batch_chunk_plan = _compute_predict_batch_chunk_plan(
            self.featurizer_info.number_of_features,
            selected_feature_count=selected_count,
            nameless_feature_count=nameless_count,
            total_pairs=num_pairs,
            total_ram_bytes=total_ram_bytes,
        )
        pair_chunk_size = max(
            1,
            min(
                int(self.batch_size),
                int(batch_chunk_plan.chunk_pairs) if batch_chunk_plan is not None else int(self.batch_size),
            ),
        )
        if batch_chunk_plan is not None:
            logger.info(
                "Predict memory chunking: total_pairs=%d chunk_pairs=%d total_ram=%d total_ram_source=%s "
                "rss=%d rss_source=%s available=%d effective_available_fraction=%.3f stage_budget_bytes=%d",
                num_pairs,
                pair_chunk_size,
                int(batch_chunk_plan.total_ram_bytes),
                str(batch_chunk_plan.total_ram_source),
                int(batch_chunk_plan.current_rss_bytes),
                str(batch_chunk_plan.current_rss_source),
                int(batch_chunk_plan.available_bytes),
                float(batch_chunk_plan.effective_available_fraction),
                int(batch_chunk_plan.stage_budget_bytes),
            )
        use_rust_blockwise = dataset_stage_uses_rust(runtime_context, dataset)
        if use_rust_blockwise:
            for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                pair_chunk_size=pair_chunk_size,
                total_ram_bytes=total_ram_bytes,
            ):
                chunk = prediction_chunk.chunk
                block_key = chunk.block_key
                if block_key != prev_block_key:
                    # cluster the completed block
                    self._flush_completed_block(
                        block_key=prev_block_key,
                        pairwise_proba=pairwise_proba,
                        block_dict=block_dict,
                        effective_cluster_model_params=effective_cluster_model_params,
                        dataset=dataset,
                        all_disallow_signature_ids=all_disallow_signature_ids,
                        pred_clusters=pred_clusters,
                    )
                    pairwise_proba = None

                    # allocate new block's matrix
                    seen_block_keys.add(block_key)
                    _guard_predict_block_matrix_allocation(
                        block_key=block_key,
                        block_size=int(chunk.block_size),
                        uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                        fastcluster_fused_dtype=fastcluster_fused_dtype,
                        total_ram_bytes=total_ram_bytes,
                    )
                    if isinstance(self.cluster_model, FastCluster):
                        pairwise_proba = np.zeros(
                            chunk.block_size * (chunk.block_size - 1) // 2,
                            dtype=fastcluster_fused_dtype,
                        )
                    else:
                        pairwise_proba = np.zeros((chunk.block_size, chunk.block_size), dtype=np.float16)

                batch_predictions = prediction_chunk.predictions
                model_predict_seconds += prediction_chunk.batch_seconds
                assert pairwise_proba is not None
                if isinstance(self.cluster_model, FastCluster):
                    start = int(chunk.start_offset)
                    end = start + int(len(chunk.labels))
                    pairwise_proba[start:end] = np.asarray(batch_predictions, dtype=pairwise_proba.dtype)
                else:
                    pairwise_proba[chunk.index_i, chunk.index_j] = np.asarray(
                        batch_predictions,
                        dtype=pairwise_proba.dtype,
                    )
                prev_block_key = block_key

            # cluster the final block
            self._flush_completed_block(
                block_key=prev_block_key,
                pairwise_proba=pairwise_proba,
                block_dict=block_dict,
                effective_cluster_model_params=effective_cluster_model_params,
                dataset=dataset,
                all_disallow_signature_ids=all_disallow_signature_ids,
                pred_clusters=pred_clusters,
            )
        else:
            block_pair_index = 0

            def _on_block_start(block_key: str) -> None:
                nonlocal pairwise_proba, block_pair_index
                pairwise_proba = None
                seen_block_keys.add(block_key)
                block_size = len(block_dict[block_key])
                _guard_predict_block_matrix_allocation(
                    block_key=block_key,
                    block_size=block_size,
                    uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                    fastcluster_fused_dtype=fastcluster_fused_dtype,
                    total_ram_bytes=total_ram_bytes,
                )
                if isinstance(self.cluster_model, FastCluster):
                    pairwise_proba = np.zeros(
                        block_size * (block_size - 1) // 2,
                        dtype=fastcluster_fused_dtype,
                    )
                else:
                    pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
                block_pair_index = 0

            def _write_prediction(
                _block_key: str,
                index_pair: tuple[int, int],
                prediction: float,
            ) -> None:
                nonlocal block_pair_index
                assert pairwise_proba is not None
                if isinstance(self.cluster_model, FastCluster):
                    pairwise_proba[block_pair_index] = prediction
                else:
                    i, j = index_pair
                    pairwise_proba[i, j] = prediction
                block_pair_index += 1

            def _post_block_callback(block_key: str) -> None:
                self._flush_completed_block(
                    block_key=block_key,
                    pairwise_proba=pairwise_proba,
                    block_dict=block_dict,
                    effective_cluster_model_params=effective_cluster_model_params,
                    dataset=dataset,
                    all_disallow_signature_ids=all_disallow_signature_ids,
                    pred_clusters=pred_clusters,
                )

            model_predict_seconds += self._featurize_predict_write_batches(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                num_pairs=num_pairs,
                write_prediction=_write_prediction,
                on_block_start=_on_block_start,
                post_block_callback=_post_block_callback,
                pair_chunk_size=pair_chunk_size,
                total_ram_bytes=total_ram_bytes,
            )

        # handle singleton blocks (0 or 1 signature — never appeared in generator)
        for block_key in block_dict.keys():
            if block_key not in seen_block_keys:
                for signature in block_dict[block_key]:
                    pred_clusters[block_key + "_0"].append(signature)

        logger.info(
            "Telemetry stage: stage=model_predict_total seconds=%.3f blocks=%d",
            model_predict_seconds,
            len(block_dict),
        )
        return dict(pred_clusters), None

    def _build_incremental_seed_setup(
        self,
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
        arrow_paths: Mapping[str, Any] | None = None,
        cluster_seed_disallows: set[tuple[str, str]] | None = None,
    ) -> tuple[
        dict[str, int | str],
        dict[int | str, int | str],
        dict[int | str, list[str]],
        dict[int | str, list[str]],
    ]:
        seed_setup_start = time.perf_counter()
        altered_presplit_predict_seconds = 0.0
        altered_presplit_block_count = 0
        altered_presplit_signature_count = 0
        altered_presplit_cache_hit_count = 0
        altered_presplit_cache_miss_count = 0
        altered_presplit_orcid_skip_count = 0
        altered_cluster_count = 0
        recluster_map: dict[int | str, int | str] = {}
        cluster_seeds_require: dict[str, int | str] = {}
        source_cluster_seeds_require = copy.deepcopy(getattr(dataset, "cluster_seeds_require", {}) or {})
        source_cluster_seeds_origin = (
            str(getattr(dataset, "_cluster_seeds_source", "dataset")) if source_cluster_seeds_require else "empty"
        )
        if not source_cluster_seeds_require:
            source_cluster_seeds_require = _cluster_seeds_require_from_arrow_paths(arrow_paths)
            if source_cluster_seeds_require:
                source_cluster_seeds_origin = "arrow"
        for signature_id, cluster_num in source_cluster_seeds_require.items():
            cluster_seeds_require[str(signature_id)] = str(cluster_num)
        cluster_seeds_require_inverse = _cluster_seeds_require_inverse(cluster_seeds_require)

        altered_cluster_signatures = _dataset_altered_cluster_signatures(dataset, arrow_paths)
        request_cluster_seed_disallows = (
            _cluster_seed_disallows_for_request(dataset, arrow_paths)
            if cluster_seed_disallows is None
            else set(normalize_cluster_seed_disallow_pairs(cluster_seed_disallows))
        )
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
            sorted_altered_cluster_nums = cast(list[int | str], sorted(altered_cluster_nums, key=str))
            altered_cluster_count = len(sorted_altered_cluster_nums)
            model_cache_fingerprint = _model_presplit_cache_fingerprint(self)
            name_tuples = getattr(dataset, "name_tuples", "filtered")
            if isinstance(arrow_paths, ValidatedArrowInputs):
                presplit_arrow_paths: Mapping[str, Any] | None = arrow_paths.without(
                    "cluster_seeds",
                    "cluster_seed_disallows",
                )
            elif arrow_paths is not None:
                presplit_arrow_paths = normalize_arrow_paths(
                    {
                        key: value
                        for key, value in arrow_paths.items()
                        if str(key) not in {"cluster_seeds", "cluster_seed_disallows"}
                    }
                )
            else:
                presplit_arrow_paths = None
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
                cache_key = (
                    _altered_presplit_cache_key(
                        mode="arrow",
                        altered_cluster_num=altered_cluster_num,
                        signature_ids=signature_ids_for_cluster_num,
                        effective_partial_supervision=effective_partial_supervision,
                        arrow_paths=presplit_arrow_paths,
                        name_tuples=name_tuples,
                        model_fingerprint=model_cache_fingerprint,
                    )
                    if presplit_arrow_paths is not None
                    else None
                )
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

            if presplit_jobs and presplit_arrow_paths is not None:
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
                reclustered_output, _ = self.predict_from_arrow_paths(
                    presplit_block_dict,
                    presplit_arrow_paths,
                    incremental_dont_use_cluster_seeds=True,
                    partial_supervision=presplit_partial_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                    load_name_counts=clusterer_uses_name_count_features(self),
                    name_tuples=name_tuples,
                    cluster_seeds_disallow=request_cluster_seed_disallows,
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
                    )
                    altered_presplit_predict_seconds += time.perf_counter() - presplit_start
                    reclustered_by_cluster_num[job.altered_cluster_num].extend(
                        [list(new_cluster) for new_cluster in reclustered_output.values()]
                    )

            for altered_cluster_num in sorted_altered_cluster_nums:
                new_clusters = reclustered_by_cluster_num.get(altered_cluster_num, [])
                if len(new_clusters) <= 1:
                    continue
                for i, new_cluster_of_signatures in enumerate(new_clusters):
                    new_cluster_num = str(altered_cluster_num) + f"_{i}"
                    recluster_map[new_cluster_num] = altered_cluster_num
                    for reclustered_signature_id in new_cluster_of_signatures:
                        cluster_seeds_require[reclustered_signature_id] = new_cluster_num

        self._last_incremental_seed_setup_telemetry = {
            "seed_setup_seconds": float(time.perf_counter() - seed_setup_start),
            "seed_setup_seed_signature_count": int(len(cluster_seeds_require)),
            "seed_setup_component_count": int(len({str(value) for value in cluster_seeds_require.values()})),
            "seed_setup_altered_signature_count": int(len(altered_cluster_signatures)),
            "seed_setup_altered_presplit_block_count": int(altered_presplit_block_count),
            "seed_setup_altered_presplit_signature_count": int(altered_presplit_signature_count),
            "seed_setup_altered_presplit_predict_seconds": float(altered_presplit_predict_seconds),
            "seed_setup_altered_presplit_cache_hit_count": int(altered_presplit_cache_hit_count),
            "seed_setup_altered_presplit_cache_miss_count": int(altered_presplit_cache_miss_count),
            "seed_setup_altered_presplit_orcid_skip_count": int(altered_presplit_orcid_skip_count),
            "seed_setup_recluster_map_entry_count": int(len(recluster_map)),
            "seed_setup_altered_cluster_count": int(altered_cluster_count),
            "seed_setup_cluster_seeds_source": source_cluster_seeds_origin,
            "seed_setup_cluster_seeds_from_arrow": int(source_cluster_seeds_origin == "arrow"),
        }
        split_cluster_seeds_require_inverse = _cluster_seeds_require_inverse(cluster_seeds_require)
        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, split_cluster_seeds_require_inverse

    def _convert_sum_count_to_average_distances(
        self,
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
    ) -> dict[str, dict[int | str, IncrementalDistStats]]:
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]] = defaultdict(dict)
        for signature_id, cluster_sum_count in signature_to_cluster_sum_count.items():
            for cluster_id, sum_count in cluster_sum_count.items():
                total = float(sum_count[0])
                count = int(sum_count[1])
                if count <= 0:
                    continue
                signature_to_cluster_to_average_dist[signature_id][cluster_id] = (
                    total / float(count),
                    count,
                    float(sum_count[2]),
                )
        return signature_to_cluster_to_average_dist

    def _finish_incremental_with_seed_links(
        self,
        unassigned_signature_ids: list[str],
        dataset: ANDData,
        linked_signature_to_cluster: Mapping[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
        arrow_paths: Mapping[str, Any] | None = None,
        split_cluster_seeds_require_inverse: Mapping[int | str, Sequence[str]] | None = None,
        cluster_seed_disallows: set[tuple[str, str]] | None = None,
    ) -> dict[str, list[str]]:
        """Apply supplied seed-link decisions, then recluster abstained signatures."""

        logger.info("Assigning unassigned signatures for incremental clustering")
        compatibility_cluster_seeds_require_inverse = (
            split_cluster_seeds_require_inverse
            if split_cluster_seeds_require_inverse is not None
            else cluster_seeds_require_inverse
        )
        pred_clusters = defaultdict(list)
        singleton_signatures = []
        for cluster_id, seed_signature_ids in cluster_seeds_require_inverse.items():
            for signature_id in seed_signature_ids:
                pred_clusters[f"{cluster_id}"].append(signature_id)
        for unassigned_signature in unassigned_signature_ids:
            if unassigned_signature not in linked_signature_to_cluster:
                singleton_signatures.append(unassigned_signature)
                continue

            best_cluster_id = linked_signature_to_cluster[unassigned_signature]
            compatibility_cluster_id = best_cluster_id
            # undo the altered-cluster split if applicable
            new_name_disallowed = False
            if best_cluster_id in recluster_map:
                best_cluster_id = recluster_map[best_cluster_id]

                if prevent_new_incompatibilities:
                    # restrict reclusterings that would add a new name incompatibility to the main cluster
                    main_cluster_signatures = compatibility_cluster_seeds_require_inverse.get(
                        compatibility_cluster_id,
                        cluster_seeds_require_inverse.get(best_cluster_id, []),
                    )
                    all_firsts = set(
                        _signature_first_for_rules(dataset.signatures[signature_id])
                        for signature_id in main_cluster_signatures
                    )
                    all_firsts = {first for first in all_firsts if len(first) > 1}

                    # if all existing first names are single characters, there is nothing else to check
                    if len(all_firsts) > 0:
                        first_unassigned = _signature_first_for_rules(dataset.signatures[unassigned_signature])
                        name_tuples = _name_tuples_for_incremental_rules(dataset.name_tuples)
                        match_found = False
                        for first_assigned in all_firsts:
                            if first_names_name_compatible(first_assigned, first_unassigned, name_tuples):
                                match_found = True
                                break
                        # if the candidate name is a prefix or a name alias for any existing name,
                        # we allow it to cluster. Otherwise, it was clustered with a single-character
                        # name and we don't want to allow that merge.
                        if not match_found:
                            signature = dataset.signatures[unassigned_signature]
                            first = signature.author_info_first
                            last = signature.author_info_last
                            paper_id = signature.paper_id
                            logger.info(
                                "Incremental clustering prevented a name compatibility issue from being "
                                f"added while clustering {first} {last} on {paper_id}"
                            )
                            new_name_disallowed = True

            if new_name_disallowed:
                singleton_signatures.append(unassigned_signature)
            else:
                pred_clusters[f"{best_cluster_id}"].append(unassigned_signature)

        request_cluster_seed_disallows = (
            _cluster_seed_disallows_for_request(dataset, arrow_paths)
            if cluster_seed_disallows is None
            else set(normalize_cluster_seed_disallow_pairs(cluster_seed_disallows))
        )
        if isinstance(arrow_paths, ValidatedArrowInputs):
            residual_arrow_paths: Mapping[str, Any] | None = arrow_paths.without("cluster_seed_disallows")
        elif arrow_paths is not None:
            residual_arrow_paths = {
                key: value for key, value in arrow_paths.items() if str(key) != "cluster_seed_disallows"
            }
        else:
            residual_arrow_paths = None

        residual_groups: list[list[str]] = []
        # all remaining singletons are reclustered together
        if len(singleton_signatures) > 0:
            logger.info("Clustering together the still unassigned signatures")
            new_cluster_id = _next_unused_cluster_id(
                pred_clusters,
                int(getattr(dataset, "max_seed_cluster_id", 0) or 0),
            )
            residual_groups = _residual_phase_b_first_initial_groups(
                self,
                dataset,
                singleton_signatures,
                partial_supervision,
            )
            residual_pair_count_before = len(singleton_signatures) * (len(singleton_signatures) - 1) // 2
            residual_pair_count_after = sum(len(group) * (len(group) - 1) // 2 for group in residual_groups)
            self._last_incremental_residual_phase_b_telemetry = {
                "residual_phase_b_signature_count": int(len(singleton_signatures)),
                "residual_phase_b_group_count": int(len(residual_groups)),
                "residual_phase_b_pair_count_before": int(residual_pair_count_before),
                "residual_phase_b_pair_count_after": int(residual_pair_count_after),
                "residual_phase_b_pair_count_saved": int(residual_pair_count_before - residual_pair_count_after),
            }
            logger.info(
                "Telemetry stage: stage=incremental_residual_phase_b residual_signatures=%d groups=%d "
                "pairs_before=%d pairs_after=%d",
                len(singleton_signatures),
                len(residual_groups),
                residual_pair_count_before,
                residual_pair_count_after,
            )
            for residual_group in residual_groups:
                if len(residual_group) == 1:
                    reclustered_output = {"singleton": [residual_group[0]]}
                else:
                    if residual_arrow_paths is None:
                        reclustered_output, _ = self.predict_helper(
                            {"block": residual_group},
                            dataset,
                            partial_supervision=partial_supervision,
                            runtime_context=runtime_context,
                            total_ram_bytes=total_ram_bytes,
                        )
                    else:
                        residual_partial_supervision = _partial_supervision_with_cluster_seed_disallows(
                            residual_group,
                            dataset,
                            partial_supervision,
                            cluster_seed_disallows=request_cluster_seed_disallows,
                        )
                        logger.info(
                            "Running incremental residual Phase B through Arrow/Rust paths: residual_signatures=%d",
                            len(residual_group),
                        )
                        reclustered_output, _ = self.predict_from_arrow_paths(
                            {"block": residual_group},
                            residual_arrow_paths,
                            partial_supervision=residual_partial_supervision,
                            runtime_context=runtime_context,
                            total_ram_bytes=total_ram_bytes,
                            load_name_counts=clusterer_uses_name_count_features(self),
                            name_tuples=getattr(dataset, "name_tuples", "filtered"),
                            cluster_seeds_disallow=request_cluster_seed_disallows,
                        )
                for new_cluster in reclustered_output.values():
                    new_cluster_id = _next_unused_cluster_id(pred_clusters, new_cluster_id)
                    pred_clusters[str(new_cluster_id)] = new_cluster
                    new_cluster_id += 1
        else:
            self._last_incremental_residual_phase_b_telemetry = {
                "residual_phase_b_signature_count": 0,
                "residual_phase_b_group_count": 0,
                "residual_phase_b_pair_count_before": 0,
                "residual_phase_b_pair_count_after": 0,
                "residual_phase_b_pair_count_saved": 0,
            }
        logger.info("Done. Returning incrementally predicted clusters")
        return dict(pred_clusters)

    def _run_incremental_phases_bcd(
        self,
        unassigned_signature_ids: list[str],
        dataset: ANDData,
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]],
        cluster_seeds_require: dict[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
        split_cluster_seeds_require_inverse: dict[int | str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        config = self._incremental_experiment_config()
        # NEW!
        # First cluster the unassigned signatures, then decide which resulting unassigned
        # clusters should merge with existing seeded clusters.
        logger.info("Batch clustering the unassigned signatures")
        incremental_only_clusters, _ = self.predict_helper(
            {"incremental_unassigned": unassigned_signature_ids},
            dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info(
            "Made %d clusters out of %d unassigned signatures",
            len(incremental_only_clusters),
            len(unassigned_signature_ids),
        )

        if config.precluster_broadcast_mode != "never":
            # Average over Phase A signature-to-seed distances at the pre-cluster level.
            # This is equivalent to computing average distance between each unassigned cluster
            # and each assigned cluster, then broadcasting that score back to member signatures.
            cluster_ids = sorted(set(cluster_seeds_require.values()), key=lambda cluster_id: str(cluster_id))
            for incremental_cluster_signature_ids in incremental_only_clusters.values():
                if config.precluster_broadcast_mode == "top1_consensus":
                    top1_cluster_id: int | str | None = None
                    should_broadcast = True
                    for signature in incremental_cluster_signature_ids:
                        best_cluster_id, best_score, _second_best_score = self._best_incremental_cluster(
                            signature_to_cluster_to_average_dist.get(signature, {}),
                            config=config,
                        )
                        if best_cluster_id is None or not math.isfinite(best_score):
                            should_broadcast = False
                            break
                        if top1_cluster_id is None:
                            top1_cluster_id = best_cluster_id
                        elif top1_cluster_id != best_cluster_id:
                            should_broadcast = False
                            break
                    if not should_broadcast:
                        continue
                for cluster_id in cluster_ids:
                    mean_dists = []
                    min_dists = []
                    support_count = 0
                    for signature in incremental_cluster_signature_ids:
                        cluster_entry = signature_to_cluster_to_average_dist.get(signature, {}).get(cluster_id)
                        if cluster_entry is None:
                            continue
                        if int(cluster_entry[1]) <= 0:
                            continue
                        mean_dists.append(float(cluster_entry[0]))
                        min_dists.append(float(cluster_entry[2]))
                        support_count += int(cluster_entry[1])
                    if len(mean_dists) == 0:
                        continue
                    out = (
                        float(np.mean(mean_dists)),
                        int(support_count),
                        float(min(min_dists)),
                    )
                    for signature in incremental_cluster_signature_ids:
                        signature_to_cluster_to_average_dist.setdefault(signature, {})[cluster_id] = out

        linked_signature_to_cluster: dict[str, int | str] = {}
        for unassigned_signature in unassigned_signature_ids:
            cluster_dists = signature_to_cluster_to_average_dist.get(unassigned_signature, {})
            best_cluster_id, best_dist, _second_best_dist = self._best_incremental_cluster(
                cluster_dists,
                config=config,
            )
            if best_cluster_id is not None and best_dist <= self.cluster_model.eps:
                linked_signature_to_cluster[unassigned_signature] = best_cluster_id

        return self._finish_incremental_with_seed_links(
            unassigned_signature_ids,
            dataset,
            linked_signature_to_cluster,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
            split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
        )

    def predict_incremental_from_arrow_paths(
        self,
        block_signatures: list[str],
        arrow_paths: Mapping[str, Any],
        *,
        prevent_new_incompatibilities: bool = True,
        batching_threshold: int | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        name_tuples: set[tuple[str, str]] | str | None = "filtered",
        cluster_seeds_require: Mapping[Any, Any] | None = None,
        cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
        altered_cluster_signatures: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Predict incremental clustering directly from Arrow IPC inputs through Rust.

        This is the direct Arrow public API for promoted incremental prediction.
        It does not require callers to construct an `ANDData`-shaped dataset
        object. Seed assignments may be supplied by `cluster_seeds_require` or
        by `arrow_paths["cluster_seeds"]`.
        """

        if runtime_context is None:
            runtime_context = build_runtime_context(
                "cluster_predict_incremental_from_arrow_paths",
                backend="rust",
            )
        elif not stage_uses_rust(runtime_context):
            raise ValueError("Clusterer.predict_incremental_from_arrow_paths requires a Rust runtime context")
        if partial_supervision is None:
            partial_supervision = {}
        require_name_counts_index = clusterer_uses_name_count_features(self)
        arrow_path_payload = validate_arrow_prediction_artifacts(
            arrow_paths,
            require_specter=clusterer_uses_embedding_features(self),
            require_name_counts_index=require_name_counts_index,
            expected_normalization_version=_resolve_clusterer_normalization_version(self),
            context="Clusterer.predict_incremental_from_arrow_paths",
            producer_hint=(
                "provide signatures, papers, paper_authors, model-required specter, model-required "
                "name_counts_index, raw-planner batch indexes, and a seed source through "
                "cluster_seeds_require or cluster_seeds.arrow"
            ),
        )
        _require_arrow_name_counts_index_for_clusterer(self, arrow_path_payload, context="Arrow incremental prediction")

        explicit_cluster_seeds_require = _normalize_cluster_seeds_require(cluster_seeds_require or {})
        if explicit_cluster_seeds_require:
            seed_signature_to_cluster = explicit_cluster_seeds_require
        else:
            seed_signature_to_cluster = _cluster_seeds_require_from_arrow_paths(arrow_path_payload)
        if not seed_signature_to_cluster and not _cluster_seeds_arrow_path_exists(arrow_path_payload):
            _require_incremental_seed_source(
                _ArrowIncrementalPredictionDataset(
                    arrow_paths=arrow_path_payload,
                    name_tuples=name_tuples,
                    cluster_seeds_require=explicit_cluster_seeds_require,
                    cluster_seeds_source="dataset",
                    cluster_seeds_disallow=cluster_seeds_disallow,
                    altered_cluster_signatures=altered_cluster_signatures,
                    max_seed_cluster_id=0,
                    signatures={},
                ),
                arrow_path_payload,
                context="Clusterer.predict_incremental_from_arrow_paths",
            )
        if not seed_signature_to_cluster:
            raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

        block_signature_ids = [str(signature_id) for signature_id in block_signatures]
        unassigned_signature_ids = [
            signature_id for signature_id in block_signature_ids if signature_id not in seed_signature_to_cluster
        ]
        altered_signature_ids = _altered_cluster_signatures_from_values_or_arrow(
            altered_cluster_signatures,
            arrow_path_payload,
        )
        altered_cluster_ids = {
            str(seed_signature_to_cluster[signature_id])
            for signature_id in altered_signature_ids
            if signature_id in seed_signature_to_cluster
        }
        altered_seed_signature_ids = [
            signature_id
            for signature_id, cluster_id in seed_signature_to_cluster.items()
            if str(cluster_id) in altered_cluster_ids
        ]
        metadata_signature_ids = (*unassigned_signature_ids, *altered_seed_signature_ids)
        signatures = _load_arrow_incremental_signature_info(arrow_path_payload, metadata_signature_ids)
        if not bool(getattr(self, "suppress_orcid", False)) and any(
            (signature_info := signatures.get(signature_id)) is not None
            and signature_info.author_info_orcid is not None
            and signature_info.author_info_orcid.strip()
            for signature_id in unassigned_signature_ids
        ):
            seed_orcid_signature_ids = [
                signature_id for signature_id in seed_signature_to_cluster if signature_id not in signatures
            ]
            signatures.update(
                _load_arrow_incremental_orcid_signature_info(
                    arrow_path_payload,
                    seed_orcid_signature_ids,
                )
            )
        request_dataset = _ArrowIncrementalPredictionDataset(
            arrow_paths=arrow_path_payload,
            name_tuples=name_tuples,
            cluster_seeds_require=seed_signature_to_cluster,
            cluster_seeds_source=("dataset" if explicit_cluster_seeds_require else "arrow"),
            cluster_seeds_disallow=cluster_seeds_disallow,
            altered_cluster_signatures=altered_cluster_signatures,
            max_seed_cluster_id=_seed_cluster_count_from_seed_map(seed_signature_to_cluster),
            signatures=signatures,
        )
        logger.info("Running direct Arrow promoted incremental prediction")
        incremental_result = predict_incremental_promoted_linker_from_arrow_paths(
            self,
            block_signature_ids,
            cast(ANDData, request_dataset),
            arrow_paths=arrow_path_payload,
            artifact_dir=_required_incremental_linker_artifact_dir(self),
            artifact=getattr(self, "incremental_linker_artifact", None),
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            batching_threshold=batching_threshold,
        )
        return incremental_result

    def predict_incremental(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool = True,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        """
        Predict clustering in incremental mode. This assumes that the majority of the labels are passed
        in using the cluster_seeds_require parameter of the dataset class, and skips work by simply assigning each
        unassigned signature to the closest cluster if distance is less than eps, and then separately clusters all
        the unassigned signatures that are not within eps of any existing cluster.

        Corrected, claimed profiles should be noted via the altered_cluster_signatures parameter (in ANDData).
        Then predict_incremental performs a pre-clustering step on each altered cluster to determine how
        S2AND would divide it into clusters. Mentions are incrementally added to these new subclusters,
        then reassembled to restore the complete claimed profile when S2AND returns results.

        Currently this would be useful in the following situation. We have a massive block, for which we want
        to cluster a small number of new signatures into (block size * number of new signatures should be less
        than the normal batch size).

        Note: this function was designed to work on a single block at a time.

        Parameters
        ----------
        block_signatures: List[str]
            the signature ids in the block to predict from
        dataset: ANDData
            the dataset
        prevent_new_incompatibilities: bool
            if True, prevents the addition to a cluster of new first names that are not prefix match
            or in the name pairs list, for at least one existing name in the cluster. This can happen
            if a claimed cluster has D Jones and David Jones, s2and would have split that cluster into two,
            and then s2and might add Donald Jones to the D Jones cluster, and once remerged, the resulting
            final cluster would have D Jones, David Jones, and Donald Jones.
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for promoted incremental query batching.
        Returns
        -------
        Dict: incremental clustering payload.
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental", backend="python")
        elif stage_uses_rust(runtime_context):
            raise ValueError(
                "Clusterer.predict_incremental operates on ANDData through Python; "
                "use predict_incremental_from_arrow_paths for Rust"
            )
        require_dataset_name_counts_binding_for_clusterer(
            self,
            dataset,
            context="Clusterer.predict_incremental",
        )
        if partial_supervision is None:
            partial_supervision = {}
        return self._predict_incremental_python(
            block_signatures,
            dataset,
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

    def _predict_incremental_python(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool = True,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Run classic incremental prediction through Python.

        This is not an external compatibility API. For behavior/parameters,
        refer to `predict_incremental`.
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental", backend="python")
        require_dataset_name_counts_binding_for_clusterer(
            self,
            dataset,
            context="Clusterer._predict_incremental_python",
        )
        if partial_supervision is None:
            partial_supervision = {}
        logger.info(f"Beginning incremental clustering for {len(block_signatures)} signatures...")
        (
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            split_cluster_seeds_require_inverse,
        ) = self._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info("Getting name constraints")
        all_pairs: list[tuple[str, str, float]] = []
        unassigned_signature_ids: list[str] = []
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
        )
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0, float("inf")))
        )
        assigned_signature_ids: list[str] = list(cluster_seeds_require.keys())
        pair_chunk_size = max(1, int(self.batch_size))
        constraint_telemetry = _ConstraintTelemetryAccumulator()

        def _update_signature_cluster_average(unassigned_signature: str, cluster_id: int | str, dist: float) -> None:
            previous_average, previous_count, previous_min = signature_to_cluster_to_average_dist[unassigned_signature][
                cluster_id
            ]
            signature_to_cluster_to_average_dist[unassigned_signature][cluster_id] = (
                (previous_average * previous_count + float(dist)) / (previous_count + 1),
                previous_count + 1,
                min(float(previous_min), float(dist)),
            )

        for possibly_unassigned_signature in block_signatures:
            if possibly_unassigned_signature in cluster_seeds_require:
                continue
            unassigned_signature_ids.append(possibly_unassigned_signature)

        pair_id_batch: list[tuple[str, str]] = []
        for unassigned_signature in unassigned_signature_ids:
            for assigned_signature in assigned_signature_ids:
                pair_id_batch.append((unassigned_signature, assigned_signature))
                if len(pair_id_batch) >= pair_chunk_size:
                    labels, batch_telemetry = self._resolve_constraint_batch(
                        dataset,
                        pair_id_batch,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=False,
                        constraint_backend=constraint_backend,
                    )
                    _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
                    all_pairs.extend(
                        (sig_id_1, sig_id_2, label)
                        for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=True)
                    )
                    pair_id_batch = []

        if pair_id_batch:
            labels, batch_telemetry = self._resolve_constraint_batch(
                dataset,
                pair_id_batch,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=False,
                constraint_backend=constraint_backend,
            )
            _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
            all_pairs.extend(
                (sig_id_1, sig_id_2, label) for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=True)
            )

        logger.info(
            "Telemetry: constraint_batch stage=predict_incremental_python total_pairs=%d "
            "partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            constraint_telemetry.total_pairs,
            constraint_telemetry.partial_supervision_hits,
            constraint_telemetry.unresolved_pairs,
            constraint_telemetry.rust_batch_call_count,
            constraint_telemetry.api_mode_summary,
            constraint_telemetry.elapsed_seconds,
            runtime_context.run_id,
        )

        logger.info("Featurizing pairs")
        batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
            all_pairs,
            dataset,
            self.featurizer_info,
            self.n_jobs,
            use_cache=self.use_cache,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info("Performing pairwise classification")
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)
        # Get predictions where there isn't partial supervision,
        # and fill the rest from partial supervision labels.
        batch_predictions, model_predict_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            batch_features,
            batch_labels,
            batch_nameless_features,
            "incremental",
            num_threads=self.n_jobs,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("Telemetry: model_predict_total seconds=%.3f blocks=1", model_predict_seconds)

        logger.info("Computing average distances for unassigned signatures")
        for signature_pair, dist in zip(all_pairs, batch_predictions, strict=True):
            unassigned_signature, assigned_signature, _ = signature_pair
            if assigned_signature not in cluster_seeds_require:
                continue
            cluster_id = cluster_seeds_require[assigned_signature]
            _update_signature_cluster_average(unassigned_signature, cluster_id, float(dist))

        predicted_clusters = self._run_incremental_phases_bcd(
            unassigned_signature_ids,
            dataset,
            signature_to_cluster_to_average_dist,
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
            split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
        )
        phase_b_required_bytes = len(unassigned_signature_ids) * (len(unassigned_signature_ids) - 1) // 2 * 8
        return _build_incremental_result(
            predicted_clusters,
            phase_b_mode="exact",
            phase_b_budget_bytes=phase_b_required_bytes,
            phase_b_required_bytes=phase_b_required_bytes,
        )
