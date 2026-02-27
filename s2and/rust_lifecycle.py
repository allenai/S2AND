from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from s2and.runtime import Backend

RustBuildPath = Literal["from_dataset", "from_json_paths"]


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


def build_rust_lifecycle_policy(
    *,
    backend: Backend,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    use_rust: bool,
    use_sinonym_overwrite: bool = False,
) -> RustLifecyclePolicy:
    if backend == "python":
        return PYTHON_ONLY_POLICY

    is_inference = _is_inference_mode(mode)

    rust_build_path: RustBuildPath = (
        "from_json_paths" if is_inference and has_signatures_path and has_papers_path else "from_dataset"
    )

    skip_python_paper_preprocess = bool(preprocess and use_rust and rust_build_path == "from_json_paths")
    defer_signature_ngrams_to_rust = bool(preprocess and use_rust)
    defer_signature_fields_to_rust = bool(preprocess and use_rust and not (is_inference and use_rust))

    return RustLifecyclePolicy(
        rust_build_path=rust_build_path,
        skip_python_paper_preprocess=skip_python_paper_preprocess,
        defer_signature_ngrams_to_rust=defer_signature_ngrams_to_rust,
        defer_signature_fields_to_rust=defer_signature_fields_to_rust,
    )
