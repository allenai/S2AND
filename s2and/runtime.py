from __future__ import annotations

import importlib
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from s2and import __version__

Backend = Literal["python", "rust"]


@dataclass(frozen=True)
class RuntimeContext:
    """Backend choice and trace identifier shared by one operation."""

    operation: str
    backend: Backend
    run_id: str


_CARGO_PRERELEASE = re.compile(r"([0-9]+\.[0-9]+\.[0-9]+)-(alpha|beta|rc)\.([0-9]+)")
_PEP440_PRERELEASE_TAGS = {"alpha": "a", "beta": "b", "rc": "rc"}


def _python_version_from_cargo(version: object) -> object:
    """Map the extension's Cargo pre-release spelling to PEP 440, e.g. 1.0.0-beta.1 -> 1.0.0b1."""
    # The extension reports its Cargo version, which spells pre-releases differently from its wheel.
    match = _CARGO_PRERELEASE.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        return version
    base, tag, number = match.groups()
    return f"{base}{_PEP440_PRERELEASE_TAGS[tag]}{number}"


def load_s2and_rust_extension(*, import_module: Callable[[str], Any] | None = None) -> Any:
    """Import the one supported Rust extension version or raise explicitly."""

    importer = import_module or importlib.import_module
    try:
        module = importer("s2and_rust")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("s2and_rust"):
            raise
        raise RuntimeError(f"Rust was requested, but s2and-rust=={__version__} is not importable") from exc

    found_version = getattr(module, "__version__", None)
    if _python_version_from_cargo(found_version) != __version__:
        raise RuntimeError(
            "Rust was requested, but the installed extension version does not match the pinned dependency: "
            f"required={__version__!r} found={found_version!r}"
        )
    return module


def _normalize_backend_value(value: str, *, label: str) -> Backend:
    normalized = value.strip().lower()
    if normalized not in {"python", "rust"}:
        raise ValueError(f"Invalid {label}={value!r}; expected 'python' or 'rust'")
    return normalized  # type: ignore[return-value]


def _resolve_backend(backend: Backend | None) -> Backend:
    requested: Backend = "python" if backend is None else _normalize_backend_value(backend, label="backend")
    if requested == "rust":
        load_s2and_rust_extension()
    return requested


def build_runtime_context(
    operation: str,
    *,
    backend: Backend | None = None,
    run_id: str | None = None,
) -> RuntimeContext:
    """Build a runtime context for one explicitly routed operation.

    Args:
        operation: Non-empty operation name used in the trace identifier.
        backend: ``"python"`` or ``"rust"``; ``None`` selects Python. Rust requires
            the exactly pinned ``s2and-rust`` extension.
        run_id: Optional trace identifier; one is generated when omitted.

    Returns:
        The resolved runtime context.
    """

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
    """Require an Arrow dataset whenever a dataset operation uses Rust."""

    if not stage_uses_rust(runtime_context):
        return False
    if getattr(dataset, "arrow_dataset", None) is not None:
        return True
    raise RuntimeError(
        f"Rust was requested for {runtime_context.operation!r}, but the dataset has no ArrowDataset. "
        "Build it with the Rust-training Arrow constructor or use backend='python'."
    )
