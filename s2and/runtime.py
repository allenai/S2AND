from __future__ import annotations

import importlib
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger("s2and")

Backend = Literal["python", "rust"]
RequestedBackend = Literal["python", "rust", "auto"]
RuntimeSource = Literal["S2AND_BACKEND", "default"]

_STARTUP_WARNING_EMITTED = False
_STARTUP_WARNING_LOCK = threading.Lock()

MIN_SUPPORTED_RUST_EXTENSION_VERSION = (0, 31, 0)
_CORE_REQUIRED_FEATURIZER_MARKERS = (
    "from_dataset",
    "from_json_paths",
    "signature_ids",
    "get_constraint",
    "get_constraints_matrix",
    "get_constraints_matrix_indexed",
    "featurize_pairs_matrix_indexed",
    "update_signature_name_counts",
)
_FEATURIZER_API_SCORE_MARKERS = tuple(
    marker
    for marker in _CORE_REQUIRED_FEATURIZER_MARKERS
    if marker
    in {
        "from_dataset",
        "from_json_paths",
        "signature_ids",
        "featurize_pairs_matrix_indexed",
        "update_signature_name_counts",
    }
)


@dataclass(frozen=True)
class RustRuntimeCapabilities:
    extension_importable: bool
    core_runtime_available: bool
    from_dataset_paper_preprocess_available: bool
    reason: str


@dataclass(frozen=True)
class BackendResolution:
    requested_backend: RequestedBackend | None
    resolved_backend: Backend
    source: RuntimeSource
    capability_reason: str


@dataclass(frozen=True)
class RuntimeContext:
    operation: str
    requested_backend: RequestedBackend | None
    resolved_backend: Backend
    use_rust: bool
    run_id: str
    source: RuntimeSource

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


def _rust_featurizer_api_score(module: Any) -> int:
    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        return -1
    return sum(1 for marker in _FEATURIZER_API_SCORE_MARKERS if hasattr(rust_featurizer_cls, marker))


def load_s2and_rust_extension(*, import_module: Callable[[str], Any] | None = None) -> Any | None:
    importer = import_module or importlib.import_module
    try:
        module = importer("s2and_rust")
    except Exception:
        return None

    shim_score = _rust_featurizer_api_score(module)

    # Workspace runs can resolve `s2and_rust` to a pure-Python shim while the compiled
    # extension lives in a submodule. Prefer the versioned native module when scores tie.
    candidate_module: Any | None = None
    try:
        candidate_module = importer("s2and_rust._s2and_rust")
    except Exception:
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
    if candidate_module is not None and candidate_score >= 0:
        return candidate_module
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
            from_dataset_paper_preprocess_available=False,
            reason="rust_extension_unavailable",
        )

    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        return RustRuntimeCapabilities(
            extension_importable=True,
            core_runtime_available=False,
            from_dataset_paper_preprocess_available=False,
            reason="rust_featurizer_missing",
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

    from_dataset_paper_preprocess_available = bool(
        core_runtime_available
        and getattr(
            rust_featurizer_cls,
            "SUPPORTS_FROM_DATASET_PAPER_PREPROCESS",
            False,
        )
    )

    return RustRuntimeCapabilities(
        extension_importable=True,
        core_runtime_available=core_runtime_available,
        from_dataset_paper_preprocess_available=from_dataset_paper_preprocess_available,
        reason=reason,
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
    )


def resolve_backend(*, emit_startup_warning: bool = True) -> BackendResolution:
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
        else:
            explicit_backend: Backend = "python" if requested == "python" else "rust"
            resolution = BackendResolution(
                requested_backend=explicit_backend,
                resolved_backend=explicit_backend,
                source="S2AND_BACKEND",
                capability_reason=f"explicit_{requested}",
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
    run_id: str | None = None,
    emit_startup_warning: bool = True,
) -> RuntimeContext:
    resolution = resolve_backend(emit_startup_warning=emit_startup_warning)
    if not operation:
        raise ValueError("operation must be a non-empty string")
    effective_run_id = run_id or f"{operation}-{uuid.uuid4().hex[:12]}"
    return RuntimeContext(
        operation=operation,
        requested_backend=resolution.requested_backend,
        resolved_backend=resolution.resolved_backend,
        use_rust=resolution.resolved_backend == "rust",
        run_id=effective_run_id,
        source=resolution.source,
    )


def stage_uses_rust(runtime_context: RuntimeContext) -> bool:
    """Returns whether Rust is enabled for the current runtime context."""
    return runtime_context.use_rust


def reset_runtime_warning_state_for_tests() -> None:
    global _STARTUP_WARNING_EMITTED
    with _STARTUP_WARNING_LOCK:
        _STARTUP_WARNING_EMITTED = False
