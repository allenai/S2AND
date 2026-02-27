import hashlib
import inspect
import json
import logging
import math
import os
import threading
import time
import weakref
from collections import Counter
from typing import Any

import numpy as np

from s2and.consts import CACHE_ROOT, CLUSTER_SEEDS_LOOKUP, FEATURIZER_VERSION, LARGE_DISTANCE
from s2and.data import ANDData
from s2and.ingest_contract import build_rust_json_ingest_contract
from s2and.rust_capabilities import detect_rust_runtime_capabilities, load_s2and_rust_extension
from s2and.rust_lifecycle import RustBuildPath, RustLifecyclePolicy

# Treat extension as Any for typing; it is optional.
_s2and_rust: Any | None


_s2and_rust = load_s2and_rust_extension()
s2and_rust: Any | None = _s2and_rust

logger = logging.getLogger("s2and")


class _CacheEntry:
    """Composite cache entry: featurizer + LRU counter + build count in one slot."""

    __slots__ = ("featurizer", "last_access", "build_count")

    def __init__(self, featurizer: Any, last_access: int = 0, build_count: int = 0):
        self.featurizer = featurizer
        self.last_access = last_access
        self.build_count = build_count


# Single WeakKeyDictionary eliminates desync risk between separate weak dicts.
_RUST_FEATURIZER_CACHE: "weakref.WeakKeyDictionary[ANDData, _CacheEntry]" = weakref.WeakKeyDictionary()
_RUST_FEATURIZER_CACHE_LOCK = threading.Lock()
_RUST_FEATURIZER_ACCESS_COUNTER = 0
RUST_FEATURIZER_CACHE_VERSION = 5
_ENV_TRUE_VALUES = {"1", "true", "yes"}
_RUST_BUILD_ERROR = "s2and_rust extension not built. Build with: maturin develop -m s2and_rust/Cargo.toml"
_SIGNATURE_NGRAM_MATERIALIZE_BATCH_SIZE = 2048
_RUST_FEATURIZER_CACHE_METADATA_SCHEMA_VERSION = 1
# TODO(s2and): Flip to "canonical_v2" once canonical artifacts (name counts, name tuples,
# ORCID prefix counts) are regenerated per docs/normalization_migration.md.
DEFAULT_NORMALIZATION_VERSION = "legacy_compat"
NORMALIZATION_VERSION_ENV = "S2AND_NORMALIZATION_VERSION"
ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV = "S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH"
_NAME_COUNTS_NORMALIZATION_VERSION_CACHE: dict[str, tuple[int, str | None]] = {}
_NAME_COUNTS_NORMALIZATION_VERSION_CACHE_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in _ENV_TRUE_VALUES


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


def _increment_rust_featurizer_build_count(dataset: ANDData) -> int:
    entry = _RUST_FEATURIZER_CACHE.get(dataset)
    if entry is not None:
        entry.build_count += 1
        return entry.build_count
    # Entry not yet inserted — caller will set build_count on insertion.
    return 1


def _rust_featurizer_build_count(dataset: ANDData) -> int:
    entry = _RUST_FEATURIZER_CACHE.get(dataset)
    return entry.build_count if entry is not None else 0


def _dataset_rust_lifecycle_policy(dataset: Any) -> RustLifecyclePolicy | None:
    policy = getattr(dataset, "rust_lifecycle_policy", None)
    if isinstance(policy, RustLifecyclePolicy):
        return policy
    return None


_RUST_FEATURIZER_MAX_INMEM_CACHE: int | None = None  # None = not yet read


def _rust_featurizer_max_inmem(dataset: ANDData | None = None) -> int:
    """Return the max number of Rust featurizers kept in memory.

    Default ``0`` (unbounded) in all modes — matching the Python feature
    cache which also keeps everything in memory for experiment iteration.
    Override with ``S2AND_RUST_FEATURIZER_MAX_INMEM=<int>`` to cap.
    """
    del dataset
    global _RUST_FEATURIZER_MAX_INMEM_CACHE
    if _RUST_FEATURIZER_MAX_INMEM_CACHE is None:
        configured = os.environ.get("S2AND_RUST_FEATURIZER_MAX_INMEM")
        if configured is not None:
            try:
                _RUST_FEATURIZER_MAX_INMEM_CACHE = max(0, int(configured))
            except ValueError:
                logger.warning(
                    "Invalid S2AND_RUST_FEATURIZER_MAX_INMEM=%s; using default 0 (unbounded)",
                    configured,
                )
                _RUST_FEATURIZER_MAX_INMEM_CACHE = 0
        else:
            _RUST_FEATURIZER_MAX_INMEM_CACHE = 0
    return _RUST_FEATURIZER_MAX_INMEM_CACHE


def _touch_rust_featurizer(dataset: ANDData) -> None:
    global _RUST_FEATURIZER_ACCESS_COUNTER
    _RUST_FEATURIZER_ACCESS_COUNTER += 1
    entry = _RUST_FEATURIZER_CACHE.get(dataset)
    if entry is not None:
        entry.last_access = _RUST_FEATURIZER_ACCESS_COUNTER


def _auto_evict_rust_featurizers(dataset_for_policy: ANDData | None = None, reserve: int = 0) -> None:
    max_inmem = _rust_featurizer_max_inmem(dataset_for_policy)
    if max_inmem <= 0:
        return
    overflow = len(_RUST_FEATURIZER_CACHE) - max_inmem + reserve
    if overflow <= 0:
        return

    # LRU by monotonic touch counter.
    # Snapshot keys to a list to guard against WeakKeyDictionary mutation
    # if garbage collection triggers a weak-ref callback during iteration.
    lru_entries = sorted(
        list(_RUST_FEATURIZER_CACHE.keys()),
        key=lambda ds: (_RUST_FEATURIZER_CACHE[ds].last_access if ds in _RUST_FEATURIZER_CACHE else -1),
    )
    evicted_names = []
    for dataset in lru_entries[:overflow]:
        if dataset in _RUST_FEATURIZER_CACHE:
            del _RUST_FEATURIZER_CACHE[dataset]
        evicted_names.append(_dataset_name_for_logs(dataset))

    if evicted_names:
        logger.info(
            "Auto-evicted %d Rust featurizer(s) from in-memory cache (max=%d): %s",
            len(evicted_names),
            max_inmem,
            ", ".join(evicted_names),
        )


def _rust_cache_path(dataset: ANDData) -> str:
    cache_dir = os.path.join(str(CACHE_ROOT), "rust_featurizer")
    os.makedirs(cache_dir, exist_ok=True)
    requested_build_path = _resolve_requested_build_path(dataset, dataset_mode=_dataset_mode_for_logs(dataset))
    cache_metadata = _rust_featurizer_cache_metadata(
        dataset,
        requested_build_path=requested_build_path,
    )
    cache_identity_hash = hashlib.sha256(
        json.dumps(cache_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    key = f"{dataset.name}_v{FEATURIZER_VERSION}_cv{RUST_FEATURIZER_CACHE_VERSION}_{cache_identity_hash}"
    return os.path.join(cache_dir, f"{key}.bin")


def _rust_cache_metadata_path(cache_path: str) -> str:
    return f"{cache_path}.meta.json"


def _artifact_identity(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "present": False}
    absolute_path = os.path.abspath(path)
    try:
        stat_result = os.stat(absolute_path)
    except OSError:
        return {"path": absolute_path, "present": False}
    return {
        "path": absolute_path,
        "present": True,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _rust_featurizer_cache_metadata(dataset: ANDData, *, requested_build_path: RustBuildPath) -> dict[str, Any]:
    rust_version = getattr(s2and_rust, "__version__", None) if s2and_rust is not None else None
    return {
        "schema_version": _RUST_FEATURIZER_CACHE_METADATA_SCHEMA_VERSION,
        "cache_version": int(RUST_FEATURIZER_CACHE_VERSION),
        "rust_version": rust_version,
        "dataset_name": getattr(dataset, "name", ""),
        "dataset_mode": _dataset_mode_for_logs(dataset),
        "requested_build_path": str(requested_build_path),
        "signature_count": int(len(getattr(dataset, "signatures", {}))),
        "paper_count": int(len(getattr(dataset, "papers", {}))),
        "name_tuple_count": int(len(getattr(dataset, "name_tuples", {}))),
        "compute_reference_features": bool(getattr(dataset, "compute_reference_features", False)),
        "preprocess": bool(getattr(dataset, "preprocess", True)),
        "skip_fasttext": bool(_env_flag("S2AND_SKIP_FASTTEXT", "")),
        "normalization_version": _expected_normalization_version(),
        "allow_normalization_version_mismatch": bool(_allow_normalization_version_mismatch()),
        "artifacts": {
            "signatures_path": _artifact_identity(getattr(dataset, "signatures_path", None)),
            "papers_path": _artifact_identity(getattr(dataset, "papers_path", None)),
            "clusters_path": _artifact_identity(getattr(dataset, "clusters_path", None)),
            "cluster_seeds_path": _artifact_identity(getattr(dataset, "cluster_seeds_path", None)),
            "specter_embeddings_path": _artifact_identity(getattr(dataset, "specter_embeddings_path", None)),
            "name_counts_json_path": _artifact_identity(_rust_name_counts_artifact_path()),
        },
    }


def _load_rust_cache_metadata(cache_path: str) -> dict[str, Any] | None:
    metadata_path = _rust_cache_metadata_path(cache_path)
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_rust_cache_metadata_best_effort(cache_path: str, cache_metadata: dict[str, Any]) -> None:
    metadata_path = _rust_cache_metadata_path(cache_path)
    temp_metadata_path = f"{metadata_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(temp_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(cache_metadata, metadata_file, sort_keys=True, separators=(",", ":"))
        os.replace(temp_metadata_path, metadata_path)
    except OSError as e:
        logger.warning("Failed to write Rust featurizer cache metadata at %s: %s", metadata_path, e)
    finally:
        if os.path.exists(temp_metadata_path):
            try:
                os.remove(temp_metadata_path)
            except OSError:
                pass


def _dataset_has_missing_signature_ngrams(dataset: ANDData) -> bool:
    for signature in dataset.signatures.values():
        if signature.author_info_affiliations_n_grams is None or signature.author_info_coauthor_n_grams is None:
            return True
    return False


def _rust_name_counts_artifact_path() -> str | None:
    configured = os.environ.get("S2AND_RUST_NAME_COUNTS_JSON", "").strip()
    return configured or None


def _signature_has_materialized_name_counts(signature: Any) -> bool:
    counts = getattr(signature, "author_info_name_counts", None)
    if counts is None:
        return False
    try:
        return any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in counts)
    except TypeError:
        return True


def _signature_name_counts_overlay_payload_from_dataset(dataset: Any) -> tuple[int, int, dict[str, Any]]:
    signatures = getattr(dataset, "signatures", None)
    if signatures is None:
        return 0, 0, {}
    try:
        signatures_total = int(len(signatures))
    except Exception:
        signatures_total = 0

    payload: dict[str, Any] = {}
    try:
        for signature_id, signature in signatures.items():
            if _signature_has_materialized_name_counts(signature):
                payload[str(signature_id)] = signature
    except Exception:
        return signatures_total, 0, {}
    return signatures_total, len(payload), payload


def _expected_normalization_version() -> str:
    configured = os.environ.get(NORMALIZATION_VERSION_ENV, DEFAULT_NORMALIZATION_VERSION).strip()
    return configured or DEFAULT_NORMALIZATION_VERSION


def _allow_normalization_version_mismatch() -> bool:
    return _env_flag(ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, "0")


def _verify_name_counts_normalization_version(name_counts_path: str) -> None:
    """Fail fast when a name-count artifact normalization version mismatches expectations."""

    if not name_counts_path:
        raise RuntimeError("Missing name_counts_path")
    absolute_path = os.path.abspath(name_counts_path)
    try:
        stat_result = os.stat(absolute_path)
    except OSError as exc:
        raise RuntimeError(f"Name counts artifact not found: {absolute_path}") from exc
    mtime_ns = int(stat_result.st_mtime_ns)

    cached_version: str | None = None
    with _NAME_COUNTS_NORMALIZATION_VERSION_CACHE_LOCK:
        cached = _NAME_COUNTS_NORMALIZATION_VERSION_CACHE.get(absolute_path)
        if cached is not None and cached[0] == mtime_ns:
            cached_version = cached[1]
        else:
            try:
                with open(absolute_path, encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Failed to load name counts artifact JSON: {absolute_path}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid name counts artifact JSON payload: {absolute_path}")
            raw_version = payload.get("normalization_version")
            cached_version = str(raw_version).strip() if raw_version is not None else None
            cached_version = cached_version or None
            _NAME_COUNTS_NORMALIZATION_VERSION_CACHE[absolute_path] = (mtime_ns, cached_version)

    if cached_version is None:
        raise RuntimeError(f"Missing normalization_version in name counts artifact: {absolute_path}")

    expected = _expected_normalization_version()
    if cached_version != expected and not _allow_normalization_version_mismatch():
        raise RuntimeError(
            "Normalization version mismatch for name counts artifact: "
            f"artifact={cached_version} expected={expected} path={absolute_path}"
        )


def _from_json_paths_supports_normalization_args(rust_featurizer_cls: Any) -> bool:
    """Return True when RustFeaturizer.from_json_paths accepts normalization version args."""

    if rust_featurizer_cls is None:
        return False
    from_json_paths = getattr(rust_featurizer_cls, "from_json_paths", None)
    if from_json_paths is None:
        return False

    text_signature = getattr(from_json_paths, "__text_signature__", None)
    if isinstance(text_signature, str) and text_signature:
        if (
            "expected_normalization_version" in text_signature
            and "allow_normalization_version_mismatch" in text_signature
        ):
            return True

    try:
        signature = inspect.signature(from_json_paths)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        doc = getattr(from_json_paths, "__doc__", None)
        if isinstance(doc, str):
            if "expected_normalization_version" in doc and "allow_normalization_version_mismatch" in doc:
                return True
        return False

    if (
        "expected_normalization_version" in signature.parameters
        and "allow_normalization_version_mismatch" in signature.parameters
    ):
        return True

    params = list(signature.parameters.values())
    if any(param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for param in params):
        # Avoid false positives when the signature is not introspectable.
        return False

    positional_params = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    # Legacy contract has 12 positional args; normalization version adds 2.
    return len(positional_params) >= 14


def _build_rust_featurizer_from_json_paths(
    dataset: Any,
    num_threads: int,
    *,
    signature_name_counts_payload: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    pre_build_start = time.perf_counter()
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    capabilities = detect_rust_runtime_capabilities(extension_module=s2and_rust)
    if not capabilities.core_runtime_available:
        raise RuntimeError(f"Rust runtime unavailable: {capabilities.reason}")
    rust_featurizer_cls = getattr(s2and_rust, "RustFeaturizer", None)
    if rust_featurizer_cls is None or not hasattr(rust_featurizer_cls, "from_json_paths"):
        raise RuntimeError("s2and_rust extension does not expose RustFeaturizer.from_json_paths")

    signatures_path = getattr(dataset, "signatures_path", None)
    papers_path = getattr(dataset, "papers_path", None)
    if not signatures_path or not papers_path:
        raise RuntimeError("Dataset does not expose signatures_path/papers_path for Rust JSON ingest")

    rust_can_overlay_signature_counts = hasattr(rust_featurizer_cls, "update_signature_name_counts")
    if signature_name_counts_payload is not None:
        dataset_signature_counts_payload = {
            str(signature_id): signature
            for signature_id, signature in signature_name_counts_payload.items()
            if _signature_has_materialized_name_counts(signature)
        }
        signatures_with_counts = len(dataset_signature_counts_payload)
        signatures_total = signatures_with_counts
        try:
            signatures_total = int(len(getattr(dataset, "signatures", {})))
        except Exception:
            pass
    else:
        signatures_total, signatures_with_counts, dataset_signature_counts_payload = (
            _signature_name_counts_overlay_payload_from_dataset(dataset)
        )
    dataset_has_signature_counts = signatures_with_counts > 0
    configured_name_counts_path = _rust_name_counts_artifact_path()
    artifact_configured = configured_name_counts_path is not None
    use_dataset_signature_counts = dataset_has_signature_counts and rust_can_overlay_signature_counts
    name_counts_source = "none"
    name_counts_path = None
    if use_dataset_signature_counts:
        name_counts_source = "dataset"
    elif not dataset_has_signature_counts and artifact_configured:
        name_counts_source = "artifact"
        name_counts_path = configured_name_counts_path
    elif dataset_has_signature_counts and not rust_can_overlay_signature_counts:
        logger.warning(
            "Rust JSON ingest: extension lacks update_signature_name_counts; "
            "cannot use precomputed signature name counts from dataset. "
            "name-count features will be NaN."
        )
    elif not dataset_has_signature_counts:
        logger.warning(
            "Rust JSON ingest: no signature name counts available and no name-count artifact selected; "
            "name-count features will be NaN."
        )

    supports_normalization_args = _from_json_paths_supports_normalization_args(rust_featurizer_cls)

    # Normalization version validation can be delegated to Rust (newer API) to avoid
    # a redundant Python json.load(); fall back to Python validation for older APIs.
    normalization_check_executed = False
    normalization_version_for_rust: str | None = None
    allow_mismatch_for_rust = False
    if name_counts_source == "artifact":
        normalization_check_executed = True
        logger.warning(
            "Rust JSON ingest: selected artifact name-count source path=%s; this can increase latency/RSS.",
            name_counts_path,
        )
        if supports_normalization_args:
            normalization_version_for_rust = _expected_normalization_version()
            allow_mismatch_for_rust = _allow_normalization_version_mismatch()
        else:
            # Older Rust APIs cannot receive expected_normalization_version; validate in Python.
            _verify_name_counts_normalization_version(str(name_counts_path))

    logger.info(
        "Telemetry stage: stage=rust_json_ingest_name_counts_source "
        "name_counts_source=%s signatures_total=%d signatures_with_counts=%d "
        "overlay_api_available=%s artifact_configured=%s "
        "normalization_check_executed=%s normalization_version_delegated_to_rust=%s "
        "allow_normalization_version_mismatch=%s dataset=%s",
        name_counts_source,
        signatures_total,
        signatures_with_counts,
        rust_can_overlay_signature_counts,
        artifact_configured,
        normalization_check_executed,
        normalization_version_for_rust is not None,
        allow_mismatch_for_rust,
        _dataset_name_for_logs(dataset),
    )

    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_start = time.perf_counter()
    contract = build_rust_json_ingest_contract(
        dataset,
        name_counts_path=name_counts_path,
        cluster_seed_require_value=CLUSTER_SEEDS_LOOKUP["require"],
        cluster_seed_disallow_value=CLUSTER_SEEDS_LOOKUP["disallow"],
        num_threads=num_threads,
        name_tuples_path=None,
        expected_normalization_version=normalization_version_for_rust,
        allow_normalization_version_mismatch=allow_mismatch_for_rust,
    )
    args = contract.as_from_json_paths_args()
    if supports_normalization_args:
        featurizer = rust_featurizer_cls.from_json_paths(*args)
    else:
        # Older Rust APIs do not accept the final normalization-version args.
        featurizer = rust_featurizer_cls.from_json_paths(*args[:-2])
    ffi_seconds = time.perf_counter() - ffi_start

    post_build_seconds = 0.0
    if name_counts_source == "dataset":
        overlay_start = time.perf_counter()
        updated_count = featurizer.update_signature_name_counts(dataset_signature_counts_payload)
        post_build_seconds = time.perf_counter() - overlay_start
        logger.info(
            "Telemetry stage: stage=rust_json_signature_name_counts_overlay "
            "seconds=%.3f updated_signatures=%d dataset=%s",
            post_build_seconds,
            int(updated_count),
            _dataset_name_for_logs(dataset),
        )
    elif name_counts_source == "none":
        logger.warning(
            "Rust JSON ingest: using name_counts_source=none; name-count features will be NaN. "
            "signatures_total=%d signatures_with_counts=%d",
            signatures_total,
            signatures_with_counts,
        )

    return featurizer, {
        "pre_build_seconds": pre_build_seconds,
        "ffi_seconds": ffi_seconds,
        "post_build_seconds": post_build_seconds,
    }


def _build_rust_featurizer_from_dataset(
    dataset: ANDData,
    *,
    rust_build_path: RustBuildPath,
) -> tuple[Any, RustBuildPath, dict[str, float]]:
    pre_build_start = time.perf_counter()
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    capabilities = detect_rust_runtime_capabilities(extension_module=s2and_rust)
    if not capabilities.core_runtime_available:
        raise RuntimeError(f"Rust runtime unavailable: {capabilities.reason}")
    num_threads = max(1, int(getattr(dataset, "n_jobs", 1)))
    selected_build_path = rust_build_path
    signatures_path = getattr(dataset, "signatures_path", None)
    papers_path = getattr(dataset, "papers_path", None)
    if selected_build_path == "from_json_paths":
        if signatures_path and papers_path:
            outer_pre_build_seconds = time.perf_counter() - pre_build_start
            featurizer, timings = _build_rust_featurizer_from_json_paths(dataset, num_threads)
            timings["pre_build_seconds"] += outer_pre_build_seconds
            return featurizer, "from_json_paths", timings
        logger.info(
            "Rust JSON ingest build path requested but unavailable; using from_dataset path "
            "(missing signatures_path/papers_path)."
        )
    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_seconds = 0.0
    ffi_start = time.perf_counter()
    featurizer = s2and_rust.RustFeaturizer.from_dataset(
        dataset,
        CLUSTER_SEEDS_LOOKUP["require"],
        CLUSTER_SEEDS_LOOKUP["disallow"],
        num_threads,
    )
    ffi_seconds += time.perf_counter() - ffi_start
    return (
        featurizer,
        "from_dataset",
        {
            "pre_build_seconds": pre_build_seconds,
            "ffi_seconds": ffi_seconds,
            "post_build_seconds": 0.0,
        },
    )


def _resolve_requested_build_path(
    dataset: ANDData,
    *,
    dataset_mode: str,
) -> RustBuildPath:
    dataset_mode_normalized = dataset_mode.strip().lower()
    has_signatures_path = bool(getattr(dataset, "signatures_path", None))
    has_papers_path = bool(getattr(dataset, "papers_path", None))
    legacy_requested_build_path: RustBuildPath = (
        "from_json_paths"
        if dataset_mode_normalized == "inference" and has_signatures_path and has_papers_path
        else "from_dataset"
    )
    rust_lifecycle_policy = _dataset_rust_lifecycle_policy(dataset)
    requested_build_path = (
        rust_lifecycle_policy.rust_build_path if rust_lifecycle_policy is not None else legacy_requested_build_path
    )
    return requested_build_path


def _try_load_rust_featurizer_from_disk_cache(
    dataset: ANDData,
    *,
    cache_path: str,
    dataset_name_for_logs: str,
    expected_cache_metadata: dict[str, Any],
) -> tuple[Any | None, str]:
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    if not os.path.exists(cache_path):
        return None, "skipped"

    cached_metadata = _load_rust_cache_metadata(cache_path)
    if cached_metadata is None:
        logger.info(
            "Telemetry: rust_core_load_cache dataset=%s disk=miss reason=metadata_missing",
            dataset_name_for_logs,
        )
        return None, "miss_metadata"
    if cached_metadata != expected_cache_metadata:
        logger.info(
            "Telemetry: rust_core_load_cache dataset=%s disk=miss reason=metadata_mismatch",
            dataset_name_for_logs,
        )
        return None, "miss_metadata"

    try:
        load_start = time.perf_counter()
        featurizer = s2and_rust.RustFeaturizer.load(cache_path)
        # Ensure cluster seeds reflect the current dataset, even if the cache is reused.
        featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
        logger.info(
            "Telemetry: rust_core_load_cache seconds=%.3f dataset=%s disk=%s",
            time.perf_counter() - load_start,
            dataset_name_for_logs,
            "hit",
        )
        return featurizer, "hit"
    except Exception as e:  # pragma: no cover - disk cache is best-effort
        logger.warning(f"Failed to load Rust featurizer cache at {cache_path}: {e}")
        return None, "attempted"


def _build_rust_featurizer_with_retry_for_missing_signature_ngrams(
    dataset: ANDData,
    *,
    requested_build_path: RustBuildPath,
) -> tuple[Any, RustBuildPath, dict[str, float], int, float]:
    build_start = time.perf_counter()
    try:
        featurizer, build_path, build_timings = _build_rust_featurizer_from_dataset(
            dataset,
            rust_build_path=requested_build_path,
        )
    except Exception as build_exc:
        missing_signature_ngrams = _dataset_has_missing_signature_ngrams(dataset)
        if missing_signature_ngrams and hasattr(dataset, "materialize_signature_ngrams_python"):
            logger.warning(
                "Rust featurizer build failed with deferred signature ngrams; "
                "materializing Python signature ngrams and retrying once: %s",
                build_exc,
            )
            try:
                dataset.materialize_signature_ngrams_python(  # type: ignore[attr-defined]
                    batch_size=_SIGNATURE_NGRAM_MATERIALIZE_BATCH_SIZE
                )
            except Exception as materialize_exc:
                logger.warning(
                    "Failed to materialize Python signature ngrams for Rust featurizer retry: %s",
                    materialize_exc,
                )
                raise
            featurizer, build_path, build_timings = _build_rust_featurizer_from_dataset(
                dataset,
                rust_build_path=requested_build_path,
            )
        else:
            raise
    build_count = _increment_rust_featurizer_build_count(dataset)
    return featurizer, build_path, build_timings, build_count, time.perf_counter() - build_start


def _save_rust_featurizer_cache_best_effort(
    featurizer: Any,
    *,
    cache_path: str,
    cache_metadata: dict[str, Any],
) -> None:
    try:
        featurizer.save(cache_path)
    except Exception as e:  # pragma: no cover - disk cache is best-effort
        logger.warning(f"Failed to save Rust featurizer cache at {cache_path}: {e}")
        return
    _write_rust_cache_metadata_best_effort(cache_path, cache_metadata)


def _get_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> Any:
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    capabilities = detect_rust_runtime_capabilities(extension_module=s2and_rust)
    if not capabilities.core_runtime_available:
        raise RuntimeError(f"Rust runtime unavailable: {capabilities.reason}")
    operation, run_id = _runtime_callsite_for_logs(dataset, runtime_context)
    dataset_mode = _dataset_mode_for_logs(dataset)
    requested_build_path = _resolve_requested_build_path(
        dataset,
        dataset_mode=dataset_mode,
    )
    save_cache_path: str | None = None
    save_cache_metadata: dict[str, Any] | None = None
    save_featurizer: Any | None = None
    featurizer: Any | None = None

    with _RUST_FEATURIZER_CACHE_LOCK:
        # Rust featurizer reuse is independent from Python pair-feature caching.
        # Disk cache still follows use_cache.
        rust_cache_enabled = True
        entry = _RUST_FEATURIZER_CACHE.get(dataset) if rust_cache_enabled else None
        cache_hit = entry is not None
        featurizer = None
        ds_log = _dataset_name_for_logs(dataset)
        cache_status = "bypass" if not rust_cache_enabled else ("hit" if cache_hit else "miss")
        logger.info(
            "Telemetry: rust_featurizer_cache cache=%s dataset=%s mode=%s op=%s run=%s builds=%d",
            cache_status,
            ds_log,
            dataset_mode,
            operation,
            run_id,
            entry.build_count if entry is not None else 0,
        )
        if entry is not None:
            _touch_rust_featurizer(dataset)
            return entry.featurizer

        use_disk_cache = bool(use_cache)
        disk_cache_status = "disabled-by-flag" if not use_disk_cache else "skipped"
        cache_path = _rust_cache_path(dataset) if use_disk_cache else None
        cache_metadata = (
            _rust_featurizer_cache_metadata(dataset, requested_build_path=requested_build_path)
            if use_disk_cache
            else None
        )
        if use_disk_cache and cache_path:
            featurizer, disk_cache_attempt_status = _try_load_rust_featurizer_from_disk_cache(
                dataset,
                cache_path=cache_path,
                dataset_name_for_logs=ds_log,
                expected_cache_metadata=cache_metadata if cache_metadata is not None else {},
            )
            if disk_cache_attempt_status != "skipped":
                disk_cache_status = disk_cache_attempt_status

        build_path = requested_build_path
        build_timings: dict[str, float] = {
            "pre_build_seconds": 0.0,
            "ffi_seconds": 0.0,
            "post_build_seconds": 0.0,
        }
        if featurizer is None:
            featurizer, build_path, build_timings, build_count, build_seconds = (
                _build_rust_featurizer_with_retry_for_missing_signature_ngrams(
                    dataset,
                    requested_build_path=requested_build_path,
                )
            )
            logger.info(
                "Telemetry: rust_core_build seconds=%.3f dataset=%s path=%s count=%d " "pre=%.3f ffi=%.3f post=%.3f",
                build_seconds,
                ds_log,
                build_path,
                build_count,
                build_timings.get("pre_build_seconds", 0.0),
                build_timings.get("ffi_seconds", 0.0),
                build_timings.get("post_build_seconds", 0.0),
            )
            if use_disk_cache and cache_path:
                # Save is intentionally deferred until after lock release.
                save_featurizer = featurizer
                save_cache_path = cache_path
                save_cache_metadata = cache_metadata
        else:
            build_count = _rust_featurizer_build_count(dataset)
        if rust_cache_enabled:
            cache_fill_source = "disk_cache" if disk_cache_status == "hit" else "build"
            _auto_evict_rust_featurizers(dataset_for_policy=dataset, reserve=1)
            global _RUST_FEATURIZER_ACCESS_COUNTER
            _RUST_FEATURIZER_ACCESS_COUNTER += 1
            _RUST_FEATURIZER_CACHE[dataset] = _CacheEntry(
                featurizer=featurizer,
                last_access=_RUST_FEATURIZER_ACCESS_COUNTER,
                build_count=build_count,
            )
            logger.info(
                "Telemetry: rust_featurizer_cache_fill source=%s disk=%s dataset=%s path=%s count=%d",
                cache_fill_source,
                disk_cache_status,
                ds_log,
                build_path,
                build_count,
            )

    if save_featurizer is not None and save_cache_path is not None and save_cache_metadata is not None:
        _save_rust_featurizer_cache_best_effort(
            save_featurizer,
            cache_path=save_cache_path,
            cache_metadata=save_cache_metadata,
        )
    if featurizer is None:
        raise RuntimeError("Rust featurizer was not initialized")
    return featurizer


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        removed = False
        if dataset in _RUST_FEATURIZER_CACHE:
            del _RUST_FEATURIZER_CACHE[dataset]
            removed = True
        return removed


def clear_rust_featurizer_cache() -> int:
    """Clear all in-memory Rust featurizer cache entries."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        count = len(_RUST_FEATURIZER_CACHE)
        _RUST_FEATURIZER_CACHE.clear()
        return count


def warm_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> None:
    """Preload the Rust featurizer into memory for low-latency inference."""
    _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)


def update_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> None:
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)
    featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)


def get_constraint_rust(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    use_cache: bool = False,
):
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)
    return featurizer.get_constraint(
        sig_id_1,
        sig_id_2,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
    )


def get_constraints_matrix_rust(
    dataset: ANDData,
    pairs: list[tuple[str, str]],
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> list[float | None]:
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)

    get_constraints_matrix = getattr(featurizer, "get_constraints_matrix", None)
    if not callable(get_constraints_matrix):
        raise RuntimeError("RustFeaturizer.get_constraints_matrix is unavailable; rebuild/install s2and-rust>=0.31.0.")
    return list(
        get_constraints_matrix(
            pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            num_threads,
        )
    )


def get_constraints_matrix_indexed_rust(
    dataset: ANDData,
    pairs: list[tuple[int, int]],
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> list[float | None]:
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)

    return list(
        featurizer.get_constraints_matrix_indexed(
            pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            num_threads,
        )
    )


def featurize_pair_rust(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    runtime_context: Any | None = None,
    use_cache: bool = False,
):
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)
    return featurizer.featurize_pair(sig_id_1, sig_id_2)


def build_pair_feature_matrix_rust(
    dataset: ANDData,
    pairs: list[tuple[str, str]],
    selected_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
    use_cache: bool = False,
) -> np.ndarray:
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)
    if not hasattr(featurizer, "featurize_pairs_matrix"):
        raise RuntimeError("RustFeaturizer.featurize_pairs_matrix is unavailable in the loaded extension")
    matrix = featurizer.featurize_pairs_matrix(
        pairs,
        selected_indices,
        num_threads,
        nan_value,
    )
    return np.asarray(matrix, dtype=np.float64)


def rust_featurizer_available() -> bool:
    capabilities = detect_rust_runtime_capabilities(extension_module=s2and_rust)
    return bool(capabilities.core_runtime_available)


def rust_signature_preprocess_available() -> bool:
    return bool(s2and_rust is not None and hasattr(s2and_rust, "signature_ngrams_batch"))


def signature_ngrams_batch_rust(
    coauthor_texts: list[str],
    affiliation_texts: list[str],
    num_threads: int | None = None,
) -> tuple[list[Counter], list[Counter]]:
    if s2and_rust is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    if not hasattr(s2and_rust, "signature_ngrams_batch"):
        raise RuntimeError("s2and_rust extension does not expose signature_ngrams_batch")
    coauthor_raw, affiliation_raw = s2and_rust.signature_ngrams_batch(
        coauthor_texts,
        affiliation_texts,
        num_threads,
    )
    if len(coauthor_raw) != len(coauthor_texts) or len(affiliation_raw) != len(affiliation_texts):
        raise RuntimeError(
            "Rust signature_ngrams_batch returned unexpected output lengths: "
            f"coauthor={len(coauthor_raw)} expected={len(coauthor_texts)} "
            f"affiliation={len(affiliation_raw)} expected={len(affiliation_texts)}"
        )
    coauthor_counters = [Counter({k: int(v) for k, v in row.items()}) for row in coauthor_raw]
    affiliation_counters = [Counter({k: int(v) for k, v in row.items()}) for row in affiliation_raw]
    return coauthor_counters, affiliation_counters
