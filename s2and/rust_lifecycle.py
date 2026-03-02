from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from s2and.runtime import Backend

RustBuildPath = Literal["from_dataset", "from_json_paths"]


@dataclass(frozen=True)
class RustJsonIngestContract:
    """Canonical inference ingest contract for RustFeaturizer.from_json_paths."""

    signatures_path: str
    papers_path: str
    clusters_path: str | None
    cluster_seeds_path: str | None
    specter_embeddings: str | dict | None
    name_tuples_path: str | None
    name_counts_path: str | None
    preprocess: bool
    compute_reference_features: bool
    cluster_seed_require_value: float
    cluster_seed_disallow_value: float
    num_threads: int
    expected_normalization_version: str | None = None
    allow_normalization_version_mismatch: bool = False

    def as_from_json_paths_args(self) -> tuple[Any, ...]:
        return (
            self.signatures_path,
            self.papers_path,
            self.cluster_seeds_path,
            self.specter_embeddings,
            self.name_tuples_path,
            self.name_counts_path,
            self.preprocess,
            self.compute_reference_features,
            self.cluster_seed_require_value,
            self.cluster_seed_disallow_value,
            self.num_threads,
            self.expected_normalization_version,
            self.allow_normalization_version_mismatch,
        )


def build_rust_json_ingest_contract(
    dataset: Any,
    *,
    name_counts_path: str | None,
    cluster_seed_require_value: float,
    cluster_seed_disallow_value: float,
    num_threads: int,
    name_tuples_path: str | None = None,
    expected_normalization_version: str | None = None,
    allow_normalization_version_mismatch: bool = False,
) -> RustJsonIngestContract:
    signatures_path = getattr(dataset, "signatures_path", None)
    papers_path = getattr(dataset, "papers_path", None)
    if not signatures_path or not papers_path:
        raise RuntimeError("Dataset does not expose signatures_path/papers_path for Rust JSON ingest")

    return RustJsonIngestContract(
        signatures_path=signatures_path,
        papers_path=papers_path,
        clusters_path=getattr(dataset, "clusters_path", None),
        cluster_seeds_path=getattr(dataset, "cluster_seeds_path", None),
        specter_embeddings=getattr(dataset, "specter_embeddings", None)
        or getattr(dataset, "specter_embeddings_path", None),
        name_tuples_path=name_tuples_path,
        name_counts_path=name_counts_path,
        preprocess=bool(getattr(dataset, "preprocess", True)),
        compute_reference_features=bool(getattr(dataset, "compute_reference_features", False)),
        cluster_seed_require_value=cluster_seed_require_value,
        cluster_seed_disallow_value=cluster_seed_disallow_value,
        num_threads=max(1, int(num_threads)),
        expected_normalization_version=expected_normalization_version,
        allow_normalization_version_mismatch=allow_normalization_version_mismatch,
    )


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen snapshot of Rust lifecycle decisions for a dataset."""

    rust_build_path: RustBuildPath
    skip_python_paper_preprocess: bool
    defer_signature_ngrams_to_rust: bool
    defer_signature_fields_to_rust: bool
    defer_rust_json_ingest_write_for_sinonym: bool


PYTHON_ONLY_POLICY = RustLifecyclePolicy(
    rust_build_path="from_dataset",
    skip_python_paper_preprocess=False,
    defer_signature_ngrams_to_rust=False,
    defer_signature_fields_to_rust=False,
    defer_rust_json_ingest_write_for_sinonym=False,
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
    compute_reference_features: bool = False,
    use_rust: bool,
    from_dataset_paper_preprocess_available: bool = False,
    use_sinonym_overwrite: bool = False,
) -> RustLifecyclePolicy:
    expected_use_rust = backend == "rust"
    if use_rust is not expected_use_rust:
        raise ValueError(
            "Inconsistent backend/use_rust configuration: "
            f"backend={backend!r} implies use_rust={expected_use_rust}, got use_rust={use_rust}."
        )

    if backend == "python":
        return PYTHON_ONLY_POLICY

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
    defer_signature_ngrams_to_rust = bool(preprocess and use_rust)
    defer_signature_fields_to_rust = bool(preprocess and use_rust and not is_inference)
    defer_rust_json_ingest_write_for_sinonym = bool(
        use_sinonym_overwrite
        and is_inference
        and rust_build_path == "from_json_paths"
        and has_signatures_path
        and has_papers_path
    )

    return RustLifecyclePolicy(
        rust_build_path=rust_build_path,
        skip_python_paper_preprocess=skip_python_paper_preprocess,
        defer_signature_ngrams_to_rust=defer_signature_ngrams_to_rust,
        defer_signature_fields_to_rust=defer_signature_fields_to_rust,
        defer_rust_json_ingest_write_for_sinonym=defer_rust_json_ingest_write_for_sinonym,
    )
