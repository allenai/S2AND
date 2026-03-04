from __future__ import annotations

import importlib
from typing import Any

from s2and import runtime as _runtime

# Backward-compatible aliases. Runtime capability ownership lives in s2and.runtime.
MIN_SUPPORTED_RUST_EXTENSION_VERSION = _runtime.MIN_SUPPORTED_RUST_EXTENSION_VERSION
RustRuntimeCapabilities = _runtime.RustRuntimeCapabilities


def load_s2and_rust_extension() -> Any | None:
    return _runtime.load_s2and_rust_extension(import_module=importlib.import_module)


def detect_rust_runtime_capabilities(extension_module: Any | None = None) -> RustRuntimeCapabilities:
    return _runtime.detect_rust_runtime_capabilities(
        extension_module=extension_module,
        import_module=importlib.import_module,
    )
