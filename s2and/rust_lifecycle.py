from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from s2and.runtime import Backend

RustBuildPath = Literal["from_dataset", "from_json_paths"]
FORCE_PYTHON_PAPER_PREPROCESS_ENV = "S2AND_RUST_FORCE_PYTHON_PAPER_PREPROCESS"


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen snapshot of Rust lifecycle decisions for a dataset."""

    rust_build_path: RustBuildPath
    skip_python_paper_preprocess: bool
    defer_signature_ngrams_to_rust: bool
    defer_signature_fields_to_rust: bool


PYTHON_ONLY_POLICY = RustLifecyclePolicy(
    rust_build_path="from_dataset",
    skip_python_paper_preprocess=False,
    defer_signature_ngrams_to_rust=False,
    defer_signature_fields_to_rust=False,
)


def _is_inference_mode(mode: str) -> bool:
    return mode.strip().lower() == "inference"


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {name}={raw_value!r}; expected one of 0/1/true/false/yes/no/on/off.")


def build_rust_lifecycle_policy(
    *,
    backend: Backend,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    compute_reference_features: bool = False,
    use_rust: bool,
    from_dataset_paper_preprocess_available: bool = False,
    use_sinonym_overwrite: bool = False,
) -> RustLifecyclePolicy:
    if backend == "python":
        return PYTHON_ONLY_POLICY

    force_python_paper_preprocess = _read_bool_env(FORCE_PYTHON_PAPER_PREPROCESS_ENV, default=False)
    is_inference = _is_inference_mode(mode)

    rust_build_path: RustBuildPath = (
        "from_json_paths" if is_inference and has_signatures_path and has_papers_path else "from_dataset"
    )

    training_from_dataset_can_skip_python_paper_preprocess = bool(
        preprocess
        and use_rust
        and not is_inference
        and rust_build_path == "from_dataset"
        and from_dataset_paper_preprocess_available
        and not compute_reference_features
    )
    skip_python_paper_preprocess = bool(
        preprocess
        and use_rust
        and (rust_build_path == "from_json_paths" or training_from_dataset_can_skip_python_paper_preprocess)
    )
    if force_python_paper_preprocess:
        skip_python_paper_preprocess = False
    defer_signature_ngrams_to_rust = bool(preprocess and use_rust)
    defer_signature_fields_to_rust = bool(preprocess and use_rust and not (is_inference and use_rust))

    return RustLifecyclePolicy(
        rust_build_path=rust_build_path,
        skip_python_paper_preprocess=skip_python_paper_preprocess,
        defer_signature_ngrams_to_rust=defer_signature_ngrams_to_rust,
        defer_signature_fields_to_rust=defer_signature_fields_to_rust,
    )
