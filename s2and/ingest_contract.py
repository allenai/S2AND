from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RustJsonIngestContract:
    """
    Canonical inference ingest contract for RustFeaturizer.from_json_paths.

    This keeps all JSON-ingest inputs in one typed payload so P2 ingest-ownership
    changes can be staged safely behind feature flags.
    """

    signatures_path: str
    papers_path: str
    clusters_path: str | None
    cluster_seeds_path: str | None
    specter_embeddings_path: str | None
    name_tuples_path: str | None
    name_counts_path: str | None
    preprocess: bool
    compute_reference_features: bool
    cluster_seed_require_value: float
    cluster_seed_disallow_value: float
    num_threads: int
    expected_normalization_version: str | None = None
    allow_normalization_version_mismatch: bool = False

    def as_from_json_paths_args(self, *, include_normalization_version_args: bool = True) -> tuple[Any, ...]:
        base_args: tuple[Any, ...] = (
            self.signatures_path,
            self.papers_path,
            self.clusters_path,
            self.cluster_seeds_path,
            self.specter_embeddings_path,
            self.name_tuples_path,
            self.name_counts_path,
            self.preprocess,
            self.compute_reference_features,
            self.cluster_seed_require_value,
            self.cluster_seed_disallow_value,
            self.num_threads,
        )
        if include_normalization_version_args:
            return base_args + (
                self.expected_normalization_version,
                self.allow_normalization_version_mismatch,
            )
        return base_args


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
        specter_embeddings_path=getattr(dataset, "specter_embeddings_path", None),
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
