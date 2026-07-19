"""Single-mode canonical normalization contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    read_name_counts_index_normalization_version,
    require_feature_contract_normalization_version,
    require_name_counts_index_artifact,
    require_normalization_version,
)
from s2and.consts import NORMALIZATION_VERSION
from s2and.model import _resolve_clusterer_normalization_version
from s2and.production_model import _require_bundle_normalization_version
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _write_minimal_name_counts_index(root: Path, *, normalization_version: str | None) -> Path:
    index_dir = root / "name_counts_index"
    data_dir = index_dir / "generations" / "gen-test"
    data_dir.mkdir(parents=True)
    files = {}
    for kind in ("first", "last", "first_last", "last_first_initial"):
        payload = data_dir / f"{kind}.bin"
        payload.write_bytes(b"")
        files[kind] = {"path": f"generations/gen-test/{kind}.bin"}
    manifest: dict = {"schema_version": "name_counts_index_v1", "files": files}
    if normalization_version is not None:
        manifest["normalization_version"] = normalization_version
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index_dir


def test_package_has_one_normalization_version() -> None:
    assert NORMALIZATION_VERSION == "canonical_v2"


@pytest.mark.parametrize("value", [None, "legacy_compat", "bogus_v9"])
def test_name_count_manifest_reader_rejects_noncanonical_versions(tmp_path: Path, value: str | None) -> None:
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=value)
    with pytest.raises(ValueError, match="invalid normalization_version"):
        read_name_counts_index_normalization_version(index_dir)


def test_name_count_manifest_reader_accepts_explicit_canonical_version(tmp_path: Path) -> None:
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=NORMALIZATION_VERSION)
    assert read_name_counts_index_normalization_version(index_dir) == NORMALIZATION_VERSION


def test_legacy_manifest_fails_artifact_validation(tmp_path: Path) -> None:
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version="legacy_compat")
    with pytest.raises(MissingArrowArtifactError, match="invalid normalization_version"):
        require_name_counts_index_artifact(index_dir, context="test", producer_hint="test")


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


@pytest.mark.parametrize("value", [None, "legacy_compat", "bogus_v9"])
def test_runtime_normalization_gate_rejects_noncanonical_versions(value: str | None) -> None:
    with pytest.raises(ValueError, match="normalization_version='canonical_v2'"):
        require_normalization_version(value, context="training")


def test_bundle_gate_accepts_canonical_and_rejects_other_versions(tmp_path: Path) -> None:
    _require_bundle_normalization_version(tmp_path, {"normalization_version": NORMALIZATION_VERSION})
    for feature_contract in ({}, {"normalization_version": "legacy_compat"}, {"normalization_version": "bogus"}):
        with pytest.raises(ValueError, match="release unit"):
            _require_bundle_normalization_version(tmp_path, feature_contract)


def test_name_counts_index_writer_stamps_package_version(tmp_path: Path) -> None:
    from s2and.incremental_linking import feature_block_arrow

    index_dir, _ = feature_block_arrow.write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
