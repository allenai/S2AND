from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from s2and.runtime import Backend, RuntimeStage

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
    stage_enablement: dict[RuntimeStage, bool],
) -> RustLifecyclePolicy:
    if backend == "python":
        return PYTHON_ONLY_POLICY

    is_inference = _is_inference_mode(mode)
    ingest_enabled = bool(stage_enablement.get("ingest_preprocess", False))
    pair_enabled = bool(stage_enablement.get("pair_featurization", False))
    constraints_enabled = bool(stage_enablement.get("constraints", False))

    rust_json_ingest_requested = is_inference and ingest_enabled
    rust_build_path: RustBuildPath = (
        "from_json_paths" if is_inference and has_signatures_path and has_papers_path else "from_dataset"
    )

    skip_python_paper_preprocess = bool(preprocess and rust_json_ingest_requested)
    defer_signature_ngrams_to_rust = bool(preprocess and ingest_enabled)
    defer_signature_fields_to_rust = bool(
        preprocess and ingest_enabled and pair_enabled and constraints_enabled and not rust_json_ingest_requested
    )

    return RustLifecyclePolicy(
        rust_build_path=rust_build_path,
        skip_python_paper_preprocess=skip_python_paper_preprocess,
        defer_signature_ngrams_to_rust=defer_signature_ngrams_to_rust,
        defer_signature_fields_to_rust=defer_signature_fields_to_rust,
    )
