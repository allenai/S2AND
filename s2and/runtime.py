from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Literal

from s2and.rust_capabilities import detect_rust_runtime_capabilities

logger = logging.getLogger("s2and")

Backend = Literal["python", "rust"]
RequestedBackend = Literal["python", "rust", "auto"]
RuntimeSource = Literal["S2AND_BACKEND", "default"]
RuntimeStage = Literal[
    "ingest_preprocess",
    "constraints",
    "pair_featurization",
]

_STARTUP_WARNING_EMITTED = False
_STARTUP_WARNING_LOCK = threading.Lock()


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
    stage_enablement: dict[RuntimeStage, bool]
    run_id: str
    source: RuntimeSource

    def stage_backend(self, stage: RuntimeStage) -> Backend:
        return "rust" if self.stage_enablement.get(stage, False) else "python"


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


def _stage_enablement_for_backend(backend: Backend) -> dict[RuntimeStage, bool]:
    if backend == "python":
        return {
            "ingest_preprocess": False,
            "constraints": False,
            "pair_featurization": False,
        }
    return {
        "ingest_preprocess": True,
        "constraints": True,
        "pair_featurization": True,
    }


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
        stage_enablement=_stage_enablement_for_backend(resolution.resolved_backend),
        run_id=effective_run_id,
        source=resolution.source,
    )


def stage_uses_rust(runtime_context: RuntimeContext, stage: RuntimeStage) -> bool:
    return runtime_context.stage_enablement.get(stage, False)


def reset_runtime_warning_state_for_tests() -> None:
    global _STARTUP_WARNING_EMITTED
    with _STARTUP_WARNING_LOCK:
        _STARTUP_WARNING_EMITTED = False
