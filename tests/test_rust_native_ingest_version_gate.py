from __future__ import annotations

import json
import os

import pytest

import s2and.feature_port as feature_port

_rust_available = feature_port.s2and_rust is not None and hasattr(
    getattr(feature_port.s2and_rust, "RustFeaturizer", None), "from_json_paths"
)
_rust_supports_normalization_args = _rust_available and feature_port._from_json_paths_supports_normalization_args(
    getattr(feature_port.s2and_rust, "RustFeaturizer", None)
)
_skip_no_rust = pytest.mark.skipif(not _rust_available, reason="s2and_rust extension unavailable")
_skip_no_normalization_args = pytest.mark.skipif(
    not _rust_supports_normalization_args,
    reason="RustFeaturizer.from_json_paths lacks normalization version args",
)


def _write_name_counts_payload(path, *, normalization_version):
    payload = {
        "first_dict": {"a": 1},
        "last_dict": {"b": 2},
        "first_last_dict": {"a b": 3},
        "last_first_initial_dict": {"b_a": 4},
    }
    if normalization_version is not None:
        payload["normalization_version"] = normalization_version
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_name_counts_version_gate_requires_metadata_by_default(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_no_version.json"
    _write_name_counts_payload(artifact_path, normalization_version=None)
    monkeypatch.setenv(feature_port.NORMALIZATION_VERSION_ENV, "legacy_compat")
    monkeypatch.delenv(feature_port.ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Missing normalization_version"):
        feature_port._verify_name_counts_normalization_version(str(artifact_path))


def test_name_counts_version_gate_fails_on_mismatch(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_mismatch.json"
    _write_name_counts_payload(artifact_path, normalization_version="canonical_v2")
    monkeypatch.setenv(feature_port.NORMALIZATION_VERSION_ENV, "legacy_compat")
    monkeypatch.delenv(feature_port.ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Normalization version mismatch"):
        feature_port._verify_name_counts_normalization_version(str(artifact_path))


def test_name_counts_version_gate_allows_explicit_override(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_override.json"
    _write_name_counts_payload(artifact_path, normalization_version="canonical_v2")
    monkeypatch.setenv(feature_port.NORMALIZATION_VERSION_ENV, "legacy_compat")
    monkeypatch.setenv(feature_port.ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, "1")

    feature_port._verify_name_counts_normalization_version(str(artifact_path))


def test_name_counts_version_gate_accepts_matching_version(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_match.json"
    _write_name_counts_payload(artifact_path, normalization_version="legacy_compat")
    monkeypatch.setenv(feature_port.NORMALIZATION_VERSION_ENV, "legacy_compat")
    monkeypatch.delenv(feature_port.ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, raising=False)

    feature_port._verify_name_counts_normalization_version(str(artifact_path))


@pytest.fixture(autouse=True)
def _cleanup_normalization_cache():
    """Clear normalization version cache before and after each test."""
    feature_port._NAME_COUNTS_NORMALIZATION_VERSION_CACHE.clear()
    yield
    feature_port._NAME_COUNTS_NORMALIZATION_VERSION_CACHE.clear()


def test_name_counts_version_gate_caches_by_path_and_mtime(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_match_cached.json"
    _write_name_counts_payload(artifact_path, normalization_version="legacy_compat")
    monkeypatch.setenv(feature_port.NORMALIZATION_VERSION_ENV, "legacy_compat")
    monkeypatch.delenv(feature_port.ALLOW_NORMALIZATION_VERSION_MISMATCH_ENV, raising=False)

    load_calls = {"count": 0}
    original_json_load = feature_port.json.load

    def _counting_json_load(file_obj):
        load_calls["count"] += 1
        return original_json_load(file_obj)

    monkeypatch.setattr(feature_port.json, "load", _counting_json_load)

    feature_port._verify_name_counts_normalization_version(str(artifact_path))
    feature_port._verify_name_counts_normalization_version(str(artifact_path))
    assert load_calls["count"] == 1

    _write_name_counts_payload(artifact_path, normalization_version="legacy_compat")
    os.utime(artifact_path, None)
    feature_port._verify_name_counts_normalization_version(str(artifact_path))
    assert load_calls["count"] == 2


# --- Rust-side normalization version validation tests ---


def _write_minimal_ingest_files(tmp_path):
    """Write minimal signatures/papers JSON for from_json_paths."""
    sigs = {
        "1": {
            "paper_id": "p1",
            "author_info": {
                "first": "A",
                "middle": None,
                "last": "B",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "a b",
                "estimated_ethnicity": None,
                "estimated_gender": None,
                "given_block": "a b",
            },
            "signature_id": "1",
            "given_name": "A B",
            "sourced_author_ids": [],
            "sourced_author_source": None,
        }
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "Test",
            "venue": "",
            "journal_name": "",
            "authors": [{"position": 0, "author_name": "A B"}],
            "references": [],
            "year": 2020,
            "abstract": "",
            "sources": [],
            "fields_of_study": [],
        }
    }
    sig_path = tmp_path / "sigs.json"
    paper_path = tmp_path / "papers.json"
    sig_path.write_text(json.dumps(sigs), encoding="utf-8")
    paper_path.write_text(json.dumps(papers), encoding="utf-8")
    return str(sig_path), str(paper_path)


def _call_from_json_paths(
    sig_path, paper_path, *, name_counts_path, expected_normalization_version, allow_normalization_version_mismatch
):
    """Call from_json_paths with positional args matching the ingest contract."""
    # Positional order: signatures, papers, clusters, cluster_seeds, specter,
    #   name_tuples, name_counts, preprocess, compute_ref, seed_require, seed_disallow,
    #   num_threads, expected_normalization_version, allow_normalization_version_mismatch
    feature_port.s2and_rust.RustFeaturizer.from_json_paths(
        sig_path,
        paper_path,
        None,  # clusters_path
        None,  # cluster_seeds_path
        None,  # specter_embeddings_path
        None,  # name_tuples_path
        name_counts_path,
        True,  # preprocess
        False,  # compute_reference_features
        0.0,  # cluster_seed_require_value
        10000.0,  # cluster_seed_disallow_value
        1,  # num_threads
        expected_normalization_version,
        allow_normalization_version_mismatch,
    )


@_skip_no_rust
@_skip_no_normalization_args
def test_rust_version_gate_fails_on_missing_version(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    _write_name_counts_payload(artifact_path, normalization_version=None)
    sig_path, paper_path = _write_minimal_ingest_files(tmp_path)

    with pytest.raises(RuntimeError, match="Missing normalization_version"):
        _call_from_json_paths(
            sig_path,
            paper_path,
            name_counts_path=str(artifact_path),
            expected_normalization_version="legacy_compat",
            allow_normalization_version_mismatch=False,
        )


@_skip_no_rust
@_skip_no_normalization_args
def test_rust_version_gate_fails_on_mismatch(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    _write_name_counts_payload(artifact_path, normalization_version="canonical_v2")
    sig_path, paper_path = _write_minimal_ingest_files(tmp_path)

    with pytest.raises(RuntimeError, match="Normalization version mismatch"):
        _call_from_json_paths(
            sig_path,
            paper_path,
            name_counts_path=str(artifact_path),
            expected_normalization_version="legacy_compat",
            allow_normalization_version_mismatch=False,
        )


@_skip_no_rust
@_skip_no_normalization_args
def test_rust_version_gate_allows_override(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    _write_name_counts_payload(artifact_path, normalization_version="canonical_v2")
    sig_path, paper_path = _write_minimal_ingest_files(tmp_path)

    # Should NOT raise when allow_normalization_version_mismatch=True
    _call_from_json_paths(
        sig_path,
        paper_path,
        name_counts_path=str(artifact_path),
        expected_normalization_version="legacy_compat",
        allow_normalization_version_mismatch=True,
    )


@_skip_no_rust
@_skip_no_normalization_args
def test_rust_version_gate_accepts_matching_version(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    _write_name_counts_payload(artifact_path, normalization_version="legacy_compat")
    sig_path, paper_path = _write_minimal_ingest_files(tmp_path)

    _call_from_json_paths(
        sig_path,
        paper_path,
        name_counts_path=str(artifact_path),
        expected_normalization_version="legacy_compat",
        allow_normalization_version_mismatch=False,
    )


@_skip_no_rust
@_skip_no_normalization_args
def test_rust_skips_validation_when_no_expected_version(tmp_path):
    """When expected_normalization_version is None, Rust skips validation."""
    artifact_path = tmp_path / "name_counts.json"
    _write_name_counts_payload(artifact_path, normalization_version="anything_goes")
    sig_path, paper_path = _write_minimal_ingest_files(tmp_path)

    # Should NOT raise — no expected version means no check
    _call_from_json_paths(
        sig_path,
        paper_path,
        name_counts_path=str(artifact_path),
        expected_normalization_version=None,
        allow_normalization_version_mismatch=False,
    )
