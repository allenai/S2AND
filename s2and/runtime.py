from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

Backend = Literal["python", "rust"]
REQUIRED_RUST_EXTENSION_VERSION = "0.60.0"


@dataclass(frozen=True)
class RuntimeContext:
    """Backend choice and trace identifier shared by one operation."""

    operation: str
    backend: Backend
    run_id: str


def load_s2and_rust_extension(*, import_module: Callable[[str], Any] | None = None) -> Any:
    """Import the one supported Rust extension version or raise explicitly."""

    importer = import_module or importlib.import_module
    try:
        module = importer("s2and_rust")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("s2and_rust"):
            raise
        raise RuntimeError(
            f"Rust was requested, but s2and-rust=={REQUIRED_RUST_EXTENSION_VERSION} is not importable"
        ) from exc

    found_version = getattr(module, "__version__", None)
    if found_version != REQUIRED_RUST_EXTENSION_VERSION:
        raise RuntimeError(
            "Rust was requested, but the installed extension version does not match the pinned dependency: "
            f"required={REQUIRED_RUST_EXTENSION_VERSION!r} found={found_version!r}"
        )
    return module


def _normalize_backend_value(value: str, *, label: str) -> Backend:
    normalized = value.strip().lower()
    if normalized not in {"python", "rust"}:
        raise ValueError(f"Invalid {label}={value!r}; expected 'python' or 'rust'")
    return normalized  # type: ignore[return-value]


def _resolve_backend(backend: Backend | None) -> Backend:
    if backend is not None:
        requested = _normalize_backend_value(backend, label="backend")
    else:
        requested_raw = os.environ.get("S2AND_BACKEND")
        if requested_raw is None:
            return "python"
        requested = _normalize_backend_value(requested_raw, label="S2AND_BACKEND")

    if requested == "rust":
        load_s2and_rust_extension()
    return requested


def build_runtime_context(
    operation: str,
    *,
    backend: Backend | None = None,
    run_id: str | None = None,
) -> RuntimeContext:
    """Build a runtime context for one explicitly routed operation."""

    if not operation:
        raise ValueError("operation must be a non-empty string")
    resolved_backend = _resolve_backend(backend)
    return RuntimeContext(
        operation=operation,
        backend=resolved_backend,
        run_id=run_id or f"{operation}-{uuid.uuid4().hex[:12]}",
    )


def stage_uses_rust(runtime_context: RuntimeContext) -> bool:
    """Return whether this explicitly resolved context uses Rust."""

    return runtime_context.backend == "rust"


def dataset_stage_uses_rust(runtime_context: RuntimeContext, dataset: Any) -> bool:
    """Require Arrow paths whenever a dataset operation explicitly uses Rust."""

    if not stage_uses_rust(runtime_context):
        return False
    if getattr(dataset, "arrow_paths", None):
        return True
    raise RuntimeError(
        f"Rust was requested for {runtime_context.operation!r}, but the dataset has no Arrow artifacts. "
        "Build it with the Rust-training Arrow constructor or use backend='python'."
    )
