"""Outcome tests for model-to-name-count manifest identity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from s2and.arrow_inputs import ValidatedArrowInputs
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer,
    require_dataset_name_counts_binding_for_clusterer,
    require_rust_featurizer_name_counts_binding_for_clusterer,
)
from s2and.name_count_binding import NameCountsBinding
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _runtime_state(tmp_path: Path):
    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    index, manifest = NameCountsIndex._open_with_manifest(index_path, context="test")
    contract = {"name_counts_manifest_sha256": manifest.manifest_sha256}
    clusterer = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=("name_counts",)),
        nameless_featurizer_info=None,
        feature_contract=contract,
    )
    dataset = SimpleNamespace(name_counts_provenance=index.source_provenance)
    arrow_paths = ValidatedArrowInputs._from_verified(
        paths={"name_counts_index": str(Path(index_path).resolve())},
        generation_id="test-generation",
        normalization_version=NORMALIZATION_VERSION,
        name_counts_manifest=manifest,
        name_counts_index=index,
    )
    return clusterer, dataset, arrow_paths, index


def test_feature_contract_contains_only_manifest_sha256(tmp_path: Path) -> None:
    clusterer, _dataset, _arrow_paths, index = _runtime_state(tmp_path)

    binding = NameCountsBinding.from_provenance(index.source_provenance, context="test index")

    assert binding.feature_contract_fields() == {
        "name_counts_manifest_sha256": index.manifest_sha256,
    }
    assert clusterer.feature_contract == binding.feature_contract_fields()


def test_exact_manifest_identity_is_accepted_at_every_runtime_boundary(tmp_path: Path) -> None:
    clusterer, dataset, arrow_paths, index = _runtime_state(tmp_path)

    require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset")
    require_arrow_name_counts_index_for_clusterer(clusterer, arrow_paths, context="Arrow")
    require_rust_featurizer_name_counts_binding_for_clusterer(clusterer, index._native, context="Rust")


def test_manifest_mismatch_is_rejected_at_every_runtime_boundary(tmp_path: Path) -> None:
    clusterer, dataset, arrow_paths, index = _runtime_state(tmp_path)
    clusterer.feature_contract = {"name_counts_manifest_sha256": "f" * 64}

    checks = (
        lambda: require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset"),
        lambda: require_arrow_name_counts_index_for_clusterer(clusterer, arrow_paths, context="Arrow"),
        lambda: require_rust_featurizer_name_counts_binding_for_clusterer(clusterer, index._native, context="Rust"),
    )
    for check in checks:
        with pytest.raises(ValueError, match="name-count binding mismatch"):
            check()


def test_missing_manifest_identity_is_rejected(tmp_path: Path) -> None:
    clusterer, dataset, _arrow_paths, _index = _runtime_state(tmp_path)
    clusterer.feature_contract = {"normalization_version": NORMALIZATION_VERSION}

    with pytest.raises(ValueError, match="name_counts_manifest_sha256"):
        require_dataset_name_counts_binding_for_clusterer(clusterer, dataset, context="dataset")
