"""Single-mode canonical normalization contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    require_feature_contract_normalization_version,
    require_name_counts_index_artifact,
    require_normalization_version,
)
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.model import _resolve_clusterer_normalization_version
from s2and.production_model import _require_bundle_normalization_version
from tests.helpers import tiny_name_counts_tuple


def _write_minimal_name_counts_index(root: Path, *, normalization_version: str | None) -> Path:
    index_path, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
    )
    index_dir = Path(index_path)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if normalization_version is None:
        manifest.pop("normalization_version")
    else:
        manifest["normalization_version"] = normalization_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return index_dir


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "missing field `normalization_version`"),
        ("legacy_compat", "unsupported normalization_version"),
    ],
)
def test_noncanonical_manifest_fails_artifact_validation(
    tmp_path: Path,
    value: str | None,
    message: str,
) -> None:
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=value)
    with pytest.raises(MissingArrowArtifactError, match=message):
        require_name_counts_index_artifact(index_dir, context="test", producer_hint="test")


def test_canonical_manifest_passes_artifact_validation(tmp_path: Path) -> None:
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=NORMALIZATION_VERSION)
    assert require_name_counts_index_artifact(index_dir, context="test", producer_hint="test") == str(index_dir)


class _ContractOnly:
    def __init__(self, feature_contract):
        self.feature_contract = feature_contract


@pytest.mark.parametrize("feature_contract", [{}, {"normalization_version": "legacy_compat"}])
def test_public_arrow_contracts_reject_missing_or_legacy_model_version(feature_contract: dict) -> None:
    owner = _ContractOnly(feature_contract)
    with pytest.raises(ValueError, match="normalization_version='canonical_v2'"):
        _resolve_clusterer_normalization_version(owner)
    with pytest.raises(ValueError, match="normalization_version='canonical_v2'"):
        require_feature_contract_normalization_version(owner, context="prediction")


def test_public_arrow_contracts_accept_canonical_model_version() -> None:
    owner = _ContractOnly({"normalization_version": NORMALIZATION_VERSION})
    assert _resolve_clusterer_normalization_version(owner) == NORMALIZATION_VERSION
    assert require_feature_contract_normalization_version(owner, context="prediction") == NORMALIZATION_VERSION
    assert require_normalization_version(NORMALIZATION_VERSION, context="training") == NORMALIZATION_VERSION


def test_runtime_normalization_gate_rejects_noncanonical_version() -> None:
    with pytest.raises(ValueError, match="normalization_version='canonical_v2'"):
        require_normalization_version("legacy_compat", context="training")


def test_bundle_gate_accepts_canonical_and_rejects_other_versions(tmp_path: Path) -> None:
    _require_bundle_normalization_version(tmp_path, {"normalization_version": NORMALIZATION_VERSION})
    with pytest.raises(ValueError, match="release unit"):
        _require_bundle_normalization_version(tmp_path, {})


def test_name_counts_index_writer_stamps_package_version(tmp_path: Path) -> None:
    from s2and.incremental_linking import feature_block_arrow

    index_dir, _ = feature_block_arrow.write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
    )
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
