"""Outcome tests for model-to-name-count manifest identity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer,
    require_dataset_name_counts_binding_for_clusterer,
    require_rust_featurizer_name_counts_binding_for_clusterer,
)
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import (
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
    write_test_arrow_artifact_manifest,
)


def _runtime_state(tmp_path: Path):
    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
    )
    index = NameCountsIndex.open(index_path)
    contract = {"name_counts_manifest_sha256": index.manifest_sha256}
    clusterer = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=("name_counts",)),
        nameless_featurizer_info=None,
        feature_contract=contract,
    )
    dataset = SimpleNamespace(name_counts_manifest_sha256=index.manifest_sha256)
    arrow_root = tmp_path / "arrow"
    paths = write_minimal_arrow_prediction_bundle(arrow_root)
    paths["name_counts_index"] = str(Path(index_path).resolve())
    write_test_arrow_artifact_manifest(arrow_root, paths)
    return clusterer, dataset, arrow_root, index


def test_exact_manifest_identity_is_accepted_at_every_runtime_boundary(tmp_path: Path) -> None:
    clusterer, dataset, arrow_root, index = _runtime_state(tmp_path)

    require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset")
    with ArrowDataset.open(arrow_root, require_name_counts_index=True) as arrow_dataset:
        require_arrow_name_counts_index_for_clusterer(clusterer, arrow_dataset, context="Arrow")
    require_rust_featurizer_name_counts_binding_for_clusterer(clusterer, index._native, context="Rust")


def test_manifest_mismatch_is_rejected_at_every_runtime_boundary(tmp_path: Path) -> None:
    clusterer, dataset, arrow_root, index = _runtime_state(tmp_path)
    clusterer.feature_contract = {"name_counts_manifest_sha256": "f" * 64}

    with pytest.raises(ValueError, match="name-count binding mismatch"):
        require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset")
    with ArrowDataset.open(arrow_root, require_name_counts_index=True) as arrow_dataset:
        with pytest.raises(ValueError, match="name-count binding mismatch"):
            require_arrow_name_counts_index_for_clusterer(clusterer, arrow_dataset, context="Arrow")
    with pytest.raises(ValueError, match="name-count binding mismatch"):
        require_rust_featurizer_name_counts_binding_for_clusterer(clusterer, index._native, context="Rust")


def test_missing_manifest_identity_is_rejected(tmp_path: Path) -> None:
    clusterer, dataset, _arrow_root, _index = _runtime_state(tmp_path)
    clusterer.feature_contract = {"normalization_version": NORMALIZATION_VERSION}

    with pytest.raises(ValueError, match="name_counts_manifest_sha256"):
        require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset")
