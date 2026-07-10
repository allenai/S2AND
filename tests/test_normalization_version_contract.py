"""normalization_version fail-fast contract (docs/normalization_migration_blocked.md, OD4).

Model bundles record the normalization policy they were trained under in
feature_contract["normalization_version"]; data artifacts record theirs in the
name_counts_index/ manifest and the Arrow dataset manifest. Absent fields mean
"legacy_compat" (pre-contract artifacts). Mismatches fail fast — there is no
runtime compatibility mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    read_name_counts_index_normalization_version,
    require_name_counts_index_artifact,
    validate_arrow_prediction_artifacts,
)
from s2and.consts import (
    NORMALIZATION_VERSION,
    NORMALIZATION_VERSION_CANONICAL_V2,
    NORMALIZATION_VERSION_LEGACY_COMPAT,
    VALID_NORMALIZATION_VERSIONS,
)
from s2and.model import _resolve_clusterer_normalization_version
from s2and.production_model import _require_bundle_normalization_version
from tests.helpers import patch_name_counts_artifact


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


def test_package_normalization_version_is_valid_token():
    assert NORMALIZATION_VERSION in VALID_NORMALIZATION_VERSIONS


def test_absent_manifest_field_defaults_to_legacy_compat(tmp_path):
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=None)
    assert read_name_counts_index_normalization_version(index_dir) == NORMALIZATION_VERSION_LEGACY_COMPAT


def test_explicit_manifest_field_is_read_back(tmp_path):
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=NORMALIZATION_VERSION_CANONICAL_V2)
    assert read_name_counts_index_normalization_version(index_dir) == NORMALIZATION_VERSION_CANONICAL_V2


def test_invalid_manifest_token_is_rejected_by_reader(tmp_path):
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version="bogus_v9")
    with pytest.raises(ValueError, match="invalid normalization_version"):
        read_name_counts_index_normalization_version(index_dir)


def test_invalid_manifest_token_fails_artifact_validation(tmp_path):
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version="bogus_v9")
    with pytest.raises(MissingArrowArtifactError, match="invalid normalization_version"):
        require_name_counts_index_artifact(index_dir, context="test", producer_hint="test")


def test_prediction_validation_rejects_artifact_model_version_mismatch(tmp_path):
    index_dir = _write_minimal_name_counts_index(tmp_path, normalization_version=NORMALIZATION_VERSION_LEGACY_COMPAT)
    for key in ("signatures", "papers", "paper_authors"):
        (tmp_path / f"{key}.arrow").write_bytes(b"")
    paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
        "name_counts_index": str(index_dir),
    }
    with pytest.raises(MissingArrowArtifactError, match="normalization_version mismatch"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=True,
            expected_normalization_version=NORMALIZATION_VERSION_CANONICAL_V2,
        )
    # Matching expectation passes.
    validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=True,
        expected_normalization_version=NORMALIZATION_VERSION_LEGACY_COMPAT,
    )


class _ContractOnly:
    def __init__(self, feature_contract):
        self.feature_contract = feature_contract


def test_clusterer_resolution_defaults_and_validates():
    assert _resolve_clusterer_normalization_version(_ContractOnly({})) == NORMALIZATION_VERSION_LEGACY_COMPAT
    assert (
        _resolve_clusterer_normalization_version(
            _ContractOnly({"normalization_version": NORMALIZATION_VERSION_CANONICAL_V2})
        )
        == NORMALIZATION_VERSION_CANONICAL_V2
    )
    with pytest.raises(ValueError, match="normalization_version"):
        _resolve_clusterer_normalization_version(_ContractOnly({"normalization_version": "bogus"}))


def test_bundle_gate_accepts_matching_and_rejects_mismatching(tmp_path):
    # This package is canonical_v2: a matching bundle loads; a pre-contract
    # bundle (absent field implies legacy_compat) and an explicit legacy bundle
    # fail fast; invalid tokens fail with a distinct message.
    _require_bundle_normalization_version(tmp_path, {"normalization_version": NORMALIZATION_VERSION})
    other = (
        NORMALIZATION_VERSION_CANONICAL_V2
        if NORMALIZATION_VERSION == NORMALIZATION_VERSION_LEGACY_COMPAT
        else NORMALIZATION_VERSION_LEGACY_COMPAT
    )
    with pytest.raises(ValueError, match="release unit"):
        _require_bundle_normalization_version(tmp_path, {})
    with pytest.raises(ValueError, match="release unit"):
        _require_bundle_normalization_version(tmp_path, {"normalization_version": other})
    with pytest.raises(ValueError, match="invalid"):
        _require_bundle_normalization_version(tmp_path, {"normalization_version": "bogus"})


def test_name_counts_index_writer_stamps_package_version(tmp_path, monkeypatch):
    from s2and.incremental_linking import feature_block_arrow

    seeded = ({"anna": 2.0}, {"smith": 3.0}, {"anna smith": 2.0}, {"smith a": 2.0})
    patch_name_counts_artifact(monkeypatch, seeded)
    index_dir, _ = feature_block_arrow.write_name_counts_index(tmp_path, overwrite=True)
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
