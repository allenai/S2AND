from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any

MIN_SUPPORTED_RUST_EXTENSION_VERSION = (0, 31, 0)
_ENV_TRUE_VALUES = {"1", "true", "yes"}
_CORE_REQUIRED_FEATURIZER_MARKERS = (
    "from_dataset",
    "from_json_paths",
    "featurize_pairs_matrix_indexed",
    "update_signature_name_counts",
)


@dataclass(frozen=True)
class RustRuntimeCapabilities:
    extension_importable: bool
    core_runtime_available: bool
    reason: str


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
    api_markers = (
        "from_dataset",
        "from_json_paths",
        "signature_ids",
        "featurize_pairs_matrix_indexed",
        "update_signature_name_counts",
    )
    return sum(1 for marker in api_markers if hasattr(rust_featurizer_cls, marker))


def load_s2and_rust_extension() -> Any | None:
    try:
        module = importlib.import_module("s2and_rust")
    except Exception:
        return None

    best_module: Any | None = module if _rust_featurizer_api_score(module) >= 0 else None
    best_score = _rust_featurizer_api_score(best_module) if best_module is not None else -1

    # Workspace runs can resolve `s2and_rust` to a namespace package at repo root.
    # Probe common submodule paths for the compiled extension and prefer whichever
    # module exposes the richest RustFeaturizer API surface.
    for candidate in ("s2and_rust._s2and_rust", "s2and_rust.s2and_rust._s2and_rust", "s2and_rust.s2and_rust"):
        try:
            candidate_module = importlib.import_module(candidate)
        except Exception:
            continue
        candidate_score = _rust_featurizer_api_score(candidate_module)
        if candidate_score > best_score:
            best_module = candidate_module
            best_score = candidate_score
            continue
        if candidate_score == best_score and best_module is not None:
            best_version = _module_version_tuple(best_module)
            candidate_version = _module_version_tuple(candidate_module)
            if best_version is None and candidate_version is not None:
                best_module = candidate_module
                best_score = candidate_score
    return best_module


def detect_rust_runtime_capabilities(extension_module: Any | None = None) -> RustRuntimeCapabilities:
    module = extension_module if extension_module is not None else load_s2and_rust_extension()
    if module is None:
        return RustRuntimeCapabilities(
            extension_importable=False,
            core_runtime_available=False,
            reason="rust_extension_unavailable",
        )

    rust_featurizer_cls = getattr(module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        return RustRuntimeCapabilities(
            extension_importable=True,
            core_runtime_available=False,
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

    return RustRuntimeCapabilities(
        extension_importable=True,
        core_runtime_available=core_runtime_available,
        reason=reason,
    )
