from __future__ import annotations

import importlib
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger("s2and")

Backend = Literal["python", "rust"]
RequestedBackend = Literal["python", "rust", "auto"]
RuntimeSource = Literal["S2AND_BACKEND", "argument", "default"]

_STARTUP_WARNING_EMITTED = False
_STARTUP_WARNING_LOCK = threading.Lock()

MIN_SUPPORTED_RUST_EXTENSION_VERSION = (0, 51, 1)
_CORE_REQUIRED_FEATURIZER_MARKERS = (
    "from_arrow_paths",
    "signature_ids",
    "get_constraints_matrix_indexed",
    "featurize_pairs_matrix_indexed",
    "update_signature_name_counts",
)
_FEATURIZER_API_SCORE_MARKERS = tuple(
    marker
    for marker in _CORE_REQUIRED_FEATURIZER_MARKERS
    if marker
    in {
        "from_arrow_paths",
        "signature_ids",
        "featurize_pairs_matrix_indexed",
        "update_signature_name_counts",
    }
)
RUST_CAPABILITY_HYBRID_CENTROID_RETRIEVER_V1 = "hybrid_centroid_retriever_v1"
RUST_CAPABILITY_INDEXED_PAIR_ARRAY_FEATURIZATION_V1 = "indexed_pair_array_featurization_v1"
RUST_CAPABILITY_INCREMENTAL_LINKING_PAIR_PLAN_V1 = "incremental_linking_pair_plan_v1"
RUST_CAPABILITY_INCREMENTAL_LINKING_CONSTRAINT_ARRAYS_V1 = "incremental_linking_constraint_arrays_v1"
RUST_CAPABILITY_RAW_ARROW_QUERY_SIGNATURE_PLANNER_V1 = "raw_arrow_query_signature_planner_v1"
_REQUIRED_INCREMENTAL_PAIR_PLAN_ROW_SIGNALS = ("row_orcid_match",)
_REQUIRED_INCREMENTAL_PAIR_PLAN_KWARGS = ("query_candidate_component_keys_by_signature_id",)
_REQUIRED_RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS = (
    "from_query_signatures",
    "plan_query_signatures",
    "build_telemetry",
)
_FROM_DATASET_RUNTIME_OPERATIONS = frozenset(
    {
        "constraints",
        "dataset_build",
        "featurization_run",
        "model_predict",
        "pair_featurization",
    }
)


@dataclass(frozen=True)
class RustRuntimeCapabilities:
    extension_importable: bool
    core_runtime_available: bool
    from_dataset_available: bool
    from_dataset_paper_preprocess_available: bool
    reason: str
    named_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendResolution:
    requested_backend: RequestedBackend | None
    resolved_backend: Backend
    source: RuntimeSource
    capability_reason: str
    from_dataset_available: bool | None = None


@dataclass(frozen=True)
class RuntimeContext:
    operation: str
    requested_backend: RequestedBackend | None
    resolved_backend: Backend
    use_rust: bool
    run_id: str
    source: RuntimeSource
    from_dataset_available: bool | None = None

    def stage_backend(self) -> Backend:
        return "rust" if self.use_rust else "python"


def _parse_semver_prefix(raw_version: str | None) -> tuple[int, int, int] | None:
    if not raw_version:
        return None
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", str(raw_version))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _module_version_tuple(module: Any) -> tuple[int, int, int] | None:
    return _parse_semver_prefix(getattr(module, "__version__", None))


def _version_tuple_to_string(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def min_supported_rust_extension_version_string() -> str:
    """Return the minimum supported Rust extension version as a semver string."""

    return _version_tuple_to_string(MIN_SUPPORTED_RUST_EXTENSION_VERSION)


def _rust_featurizer_api_score(module: Any) -> int:
    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        return -1
    return sum(1 for marker in _FEATURIZER_API_SCORE_MARKERS if hasattr(rust_featurizer_cls, marker))


def _rust_build_info(module: Any) -> Mapping[str, Any]:
    get_build_info = getattr(module, "get_build_info", None)
    if not callable(get_build_info):
        return {}
    build_info = get_build_info()
    return build_info if isinstance(build_info, Mapping) else {}


def _build_info_sequence_contains_all(module: Any, key: str, required: tuple[str, ...]) -> bool:
    values = _rust_build_info(module).get(key)
    if values is None:
        return False
    if isinstance(values, str):
        available = {values}
    else:
        try:
            available = {str(value) for value in values}
        except TypeError:
            return False
    return all(value in available for value in required)


def _has_current_incremental_pair_plan_abi(module: Any) -> bool:
    if not _build_info_sequence_contains_all(
        module,
        "incremental_linking_pair_plan_supported_kwargs",
        _REQUIRED_INCREMENTAL_PAIR_PLAN_KWARGS,
    ):
        return False
    return _build_info_sequence_contains_all(
        module,
        "incremental_linking_pair_plan_row_signals",
        _REQUIRED_INCREMENTAL_PAIR_PLAN_ROW_SIGNALS,
    )


def _has_current_raw_arrow_query_signature_planner_abi(module: Any) -> bool:
    raw_planner_cls = getattr(module, "RawBlockQueryCandidatePlanner", None)
    if raw_planner_cls is None or not callable(getattr(raw_planner_cls, "from_query_signatures", None)):
        return False
    return _build_info_sequence_contains_all(
        module,
        "raw_arrow_query_signature_planner_methods",
        _REQUIRED_RAW_ARROW_QUERY_SIGNATURE_PLANNER_METHODS,
    )


def _detect_named_rust_capabilities(module: Any) -> tuple[str, ...]:
    capabilities: list[str] = []
    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    rust_retriever_cls = getattr(module, "RustHybridCentroidRetriever", None)
    if rust_retriever_cls is not None and callable(
        getattr(rust_retriever_cls, "top_k_hybrid_centroid_pair_plan", None)
    ):
        capabilities.append(RUST_CAPABILITY_HYBRID_CENTROID_RETRIEVER_V1)
    if rust_featurizer_cls is not None and callable(
        getattr(rust_featurizer_cls, "linker_pair_index_arrays_and_aggregate_stats", None)
    ):
        capabilities.append(RUST_CAPABILITY_INDEXED_PAIR_ARRAY_FEATURIZATION_V1)
    if (
        rust_retriever_cls is not None
        and callable(getattr(rust_retriever_cls, "top_k_hybrid_centroid_pair_plan", None))
        and _has_current_incremental_pair_plan_abi(module)
    ):
        capabilities.append(RUST_CAPABILITY_INCREMENTAL_LINKING_PAIR_PLAN_V1)
    if (
        rust_featurizer_cls is not None
        and callable(getattr(rust_featurizer_cls, "linker_pair_index_arrays_constraint_labels", None))
        and callable(getattr(rust_featurizer_cls, "linker_pair_distance_accumulators", None))
    ):
        capabilities.append(RUST_CAPABILITY_INCREMENTAL_LINKING_CONSTRAINT_ARRAYS_V1)
    if _has_current_raw_arrow_query_signature_planner_abi(module):
        capabilities.append(RUST_CAPABILITY_RAW_ARROW_QUERY_SIGNATURE_PLANNER_V1)
    return tuple(capabilities)


def _is_missing_s2and_rust_native_module(exc: ModuleNotFoundError) -> bool:
    """Return whether an import failure means the optional Rust extension is absent."""

    missing_name = exc.name or ""
    return missing_name == "s2and_rust" or (
        missing_name.startswith("s2and_rust.") and missing_name.endswith("._s2and_rust")
    )


def load_s2and_rust_extension(*, import_module: Callable[[str], Any] | None = None) -> Any | None:
    importer = import_module or importlib.import_module
    try:
        module = importer("s2and_rust")
    except ModuleNotFoundError as exc:
        if not _is_missing_s2and_rust_native_module(exc):
            raise
        return None

    shim_score = _rust_featurizer_api_score(module)

    # Workspace runs can resolve `s2and_rust` to a pure-Python shim while the compiled
    # extension lives in a submodule. Prefer the versioned native module when scores tie.
    candidate_module: Any | None = None
    try:
        candidate_module = importer("s2and_rust._s2and_rust")
    except ModuleNotFoundError as exc:
        if not _is_missing_s2and_rust_native_module(exc):
            raise
        candidate_module = None

    candidate_score = _rust_featurizer_api_score(candidate_module) if candidate_module is not None else -1
    if candidate_module is not None and candidate_score >= 0:
        if candidate_score > shim_score:
            return candidate_module
        if candidate_score == shim_score:
            shim_version = _module_version_tuple(module)
            candidate_version = _module_version_tuple(candidate_module)
            if candidate_version is not None and shim_version is None:
                return candidate_module
            if candidate_version is not None and shim_version is not None and candidate_version > shim_version:
                return candidate_module

    if shim_score >= 0:
        return module
    return None


def detect_rust_runtime_capabilities(
    extension_module: Any | None = None,
    *,
    import_module: Callable[[str], Any] | None = None,
) -> RustRuntimeCapabilities:
    module = (
        extension_module if extension_module is not None else load_s2and_rust_extension(import_module=import_module)
    )
    if module is None:
        return RustRuntimeCapabilities(
            extension_importable=False,
            core_runtime_available=False,
            from_dataset_available=False,
            from_dataset_paper_preprocess_available=False,
            reason="rust_extension_unavailable",
            named_capabilities=(),
        )

    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        return RustRuntimeCapabilities(
            extension_importable=True,
            core_runtime_available=False,
            from_dataset_available=False,
            from_dataset_paper_preprocess_available=False,
            reason="rust_featurizer_missing",
            named_capabilities=_detect_named_rust_capabilities(module),
        )

    missing_markers = [
        marker for marker in _CORE_REQUIRED_FEATURIZER_MARKERS if not hasattr(rust_featurizer_cls, marker)
    ]
    core_runtime_available = len(missing_markers) == 0

    if not core_runtime_available:
        reason = "rust_core_missing_markers:" + ",".join(missing_markers)
    else:
        version_tuple = _module_version_tuple(module)
        if version_tuple is None:
            core_runtime_available = False
            reason = f"rust_version_unparseable:{getattr(module, '__version__', None)!r}"
        elif version_tuple < MIN_SUPPORTED_RUST_EXTENSION_VERSION:
            core_runtime_available = False
            reason = (
                "rust_version_below_minimum:"
                f"{_version_tuple_to_string(version_tuple)}<"
                f"{_version_tuple_to_string(MIN_SUPPORTED_RUST_EXTENSION_VERSION)}"
            )
        else:
            reason = "rust_core_available"

    from_dataset_available = bool(
        core_runtime_available and callable(getattr(rust_featurizer_cls, "from_dataset", None))
    )
    from_dataset_paper_preprocess_available = bool(
        from_dataset_available
        and getattr(
            rust_featurizer_cls,
            "SUPPORTS_FROM_DATASET_PAPER_PREPROCESS",
            False,
        )
    )

    return RustRuntimeCapabilities(
        extension_importable=True,
        core_runtime_available=core_runtime_available,
        from_dataset_available=from_dataset_available,
        from_dataset_paper_preprocess_available=from_dataset_paper_preprocess_available,
        reason=reason,
        named_capabilities=_detect_named_rust_capabilities(module),
    )


def _normalize_backend_value(value: str) -> str:
    return value.strip().lower()


def _emit_startup_runtime_warning_once(resolution: BackendResolution) -> None:
    global _STARTUP_WARNING_EMITTED
    with _STARTUP_WARNING_LOCK:
        if _STARTUP_WARNING_EMITTED:
            return
        _STARTUP_WARNING_EMITTED = True

    logger.warning(
        "Runtime backend resolved at startup: resolved_backend=%s source=%s requested_backend=%s capability_reason=%s",
        resolution.resolved_backend,
        resolution.source,
        resolution.requested_backend,
        resolution.capability_reason,
    )


def _auto_backend_capability_probe() -> tuple[bool, str]:
    capabilities = detect_rust_runtime_capabilities()
    return capabilities.core_runtime_available, capabilities.reason


def _resolve_auto_backend(
    *,
    requested_backend: RequestedBackend | None,
    source: RuntimeSource,
) -> BackendResolution:
    rust_core_available, capability_reason = _auto_backend_capability_probe()
    resolved_backend: Backend = "rust" if rust_core_available else "python"
    return BackendResolution(
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        source=source,
        capability_reason=capability_reason,
        from_dataset_available=None,
    )


def _resolve_explicit_rust_backend(*, source: RuntimeSource) -> BackendResolution:
    capabilities = detect_rust_runtime_capabilities()
    if not capabilities.core_runtime_available:
        min_version = min_supported_rust_extension_version_string()
        request_label = "backend='rust'" if source == "argument" else "S2AND_BACKEND='rust'"
        raise RuntimeError(
            f"{request_label} requested but Rust runtime is unavailable or unsupported "
            f"(reason={capabilities.reason}). Install/upgrade s2and_rust (>= {min_version}) "
            "or use backend='python'/'auto'."
        )
    return BackendResolution(
        requested_backend="rust",
        resolved_backend="rust",
        source=source,
        capability_reason=capabilities.reason,
        from_dataset_available=capabilities.from_dataset_available,
    )


def resolve_backend(*, emit_startup_warning: bool = True) -> BackendResolution:
    return resolve_backend_for_request(backend=None, emit_startup_warning=emit_startup_warning)


def resolve_backend_for_request(
    *,
    backend: RequestedBackend | None = None,
    emit_startup_warning: bool = True,
) -> BackendResolution:
    if backend is not None:
        requested = _normalize_backend_value(backend)
        if requested not in {"python", "rust", "auto"}:
            raise ValueError(f"Invalid backend={backend!r}; expected 'python', 'rust', or 'auto'")
        if requested == "auto":
            resolution = _resolve_auto_backend(
                requested_backend="auto",
                source="argument",
            )
        elif requested == "rust":
            resolution = _resolve_explicit_rust_backend(source="argument")
        else:
            resolution = BackendResolution(
                requested_backend="python",
                resolved_backend="python",
                source="argument",
                capability_reason="explicit_python",
            )
        if emit_startup_warning:
            _emit_startup_runtime_warning_once(resolution)
        return resolution

    requested_raw = os.environ.get("S2AND_BACKEND")
    if requested_raw is not None:
        requested = _normalize_backend_value(requested_raw)
        if requested not in {"python", "rust", "auto"}:
            raise ValueError(f"Invalid S2AND_BACKEND={requested_raw!r}; expected 'python', 'rust', or 'auto'")
        if requested == "auto":
            resolution = _resolve_auto_backend(
                requested_backend="auto",
                source="S2AND_BACKEND",
            )
        elif requested == "rust":
            resolution = _resolve_explicit_rust_backend(source="S2AND_BACKEND")
        else:
            resolution = BackendResolution(
                requested_backend="python",
                resolved_backend="python",
                source="S2AND_BACKEND",
                capability_reason="explicit_python",
            )
        if emit_startup_warning:
            _emit_startup_runtime_warning_once(resolution)
        return resolution

    resolution = _resolve_auto_backend(
        requested_backend=None,
        source="default",
    )
    if emit_startup_warning:
        _emit_startup_runtime_warning_once(resolution)
    return resolution


def build_runtime_context(
    operation: str,
    *,
    backend: RequestedBackend | None = None,
    run_id: str | None = None,
    emit_startup_warning: bool = True,
) -> RuntimeContext:
    resolution = resolve_backend_for_request(backend=backend, emit_startup_warning=emit_startup_warning)
    if not operation:
        raise ValueError("operation must be a non-empty string")
    resolved_backend = resolution.resolved_backend
    use_rust = resolved_backend == "rust"
    from_dataset_available = resolution.from_dataset_available
    if use_rust and operation in _FROM_DATASET_RUNTIME_OPERATIONS:
        if from_dataset_available is None:
            from_dataset_available = detect_rust_runtime_capabilities().from_dataset_available
        if not from_dataset_available:
            if resolution.requested_backend == "rust":
                request_label = "backend='rust'" if resolution.source == "argument" else "S2AND_BACKEND='rust'"
                raise RuntimeError(
                    f"{request_label} requested for {operation!r}, but RustFeaturizer.from_dataset is unavailable. "
                    "Use backend='python'/'auto' for ANDData training/inference paths or use Arrow production paths."
                )
            resolved_backend = "python"
            use_rust = False
    effective_run_id = run_id or f"{operation}-{uuid.uuid4().hex[:12]}"
    return RuntimeContext(
        operation=operation,
        requested_backend=resolution.requested_backend,
        resolved_backend=resolved_backend,
        use_rust=use_rust,
        run_id=effective_run_id,
        source=resolution.source,
        from_dataset_available=from_dataset_available,
    )


def stage_uses_rust(runtime_context: RuntimeContext) -> bool:
    """Returns whether Rust is enabled for the current runtime context."""
    if not runtime_context.use_rust:
        return False
    if (
        runtime_context.operation in _FROM_DATASET_RUNTIME_OPERATIONS
        and getattr(runtime_context, "from_dataset_available", None) is False
    ):
        return False
    return True


def reset_runtime_warning_state_for_tests() -> None:
    global _STARTUP_WARNING_EMITTED
    with _STARTUP_WARNING_LOCK:
        _STARTUP_WARNING_EMITTED = False
