from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts._pair_ablation import legacy_rust


def _write_legacy_manifest(tmp_path: Path, *, child_normalization: str | None = None) -> Path:
    for filename, contents in (
        ("signatures.arrow", b"signatures"),
        ("papers.arrow", b"papers"),
        ("paper_authors.arrow", b"paper-authors"),
        ("specter1.arrow", b"specter-one"),
        ("specter2.arrow", b"specter-two"),
        ("specter2.index", b"specter-two-index"),
    ):
        (tmp_path / filename).write_bytes(contents)
    name_counts = tmp_path / "name_counts"
    name_counts.mkdir()
    (name_counts / "counts.bin").write_bytes(b"counts")
    child_manifest: dict[str, Any] = {"schema_version": "name_counts_index_v1"}
    if child_normalization is not None:
        child_manifest["normalization_version"] = child_normalization
    (name_counts / "manifest.json").write_text(json.dumps(child_manifest), encoding="utf-8")

    manifest = {
        "dataset": "tiny_legacy",
        "paths": {
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "specter": "specter1.arrow",
            "specter2": "specter2.arrow",
            "specter2_batch_index": "specter2.index",
            "name_counts_index": "name_counts",
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_resolve_legacy_manifest_explicitly_aliases_specter2_and_reports_digests(tmp_path: Path) -> None:
    manifest_path = _write_legacy_manifest(tmp_path)

    artifacts = legacy_rust.resolve_legacy_arrow_manifest(manifest_path)

    assert artifacts.dataset == "tiny_legacy"
    assert "specter2" in artifacts.source_paths
    assert "specter" not in artifacts.source_paths
    assert artifacts.rust_paths["specter"] == (tmp_path / "specter2.arrow").resolve()
    assert artifacts.rust_paths["specter_batch_index"] == (tmp_path / "specter2.index").resolve()
    assert "specter2" not in artifacts.rust_paths
    assert artifacts.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    identity = legacy_rust.current_artifact_identity(artifacts, include_path_digests=True)

    assert identity["embedding_alias"] == {
        "manifest_path_key": "specter2",
        "rust_path_key": "specter",
    }
    assert identity["path_digests"]["specter2"]["sha256"] == hashlib.sha256(b"specter-two").hexdigest()
    assert identity["path_digests"]["name_counts_index"]["kind"] == "directory"
    assert identity["path_digests"]["name_counts_index"]["size_bytes"] > len(b"counts")


def test_digest_cache_reuses_content_hashes_across_dataset_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_manifest = _write_legacy_manifest(first_dir)
    second_manifest = _write_legacy_manifest(second_dir)
    second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    second_payload["dataset"] = "second_legacy"
    second_payload["paths"]["paper_authors"] = str((first_dir / "paper_authors.arrow").resolve())
    second_payload["paths"]["name_counts_index"] = str((first_dir / "name_counts").resolve())
    second_manifest.write_text(json.dumps(second_payload, sort_keys=True), encoding="utf-8")
    first = legacy_rust.resolve_legacy_arrow_manifest(first_manifest)
    second = legacy_rust.resolve_legacy_arrow_manifest(second_manifest)

    calls: list[Path] = []
    real_digest_path = legacy_rust._digest_path

    def counting_digest_path(path: Path) -> legacy_rust.ArtifactDigest:
        calls.append(path)
        return real_digest_path(path)

    monkeypatch.setattr(legacy_rust, "_digest_path", counting_digest_path)
    digest_cache: dict[Path, legacy_rust.ArtifactDigest] = {}

    first_identity = legacy_rust.current_artifact_identity(
        first,
        include_path_digests=True,
        digest_cache=digest_cache,
    )
    second_identity = legacy_rust.current_artifact_identity(
        second,
        include_path_digests=True,
        digest_cache=digest_cache,
    )

    shared_paper_authors = (first_dir / "paper_authors.arrow").resolve()
    shared_name_counts = (first_dir / "name_counts").resolve()
    assert calls.count(shared_paper_authors) == 1
    assert calls.count(shared_name_counts) == 1
    assert len(calls) == len(set(first.source_paths.values()) | set(second.source_paths.values()))
    assert set(digest_cache) == set(first.source_paths.values()) | set(second.source_paths.values())
    assert (
        first_identity["path_digests"]["name_counts_index"]["sha256"]
        == second_identity["path_digests"]["name_counts_index"]["sha256"]
        == digest_cache[shared_name_counts].sha256
    )


def test_digest_cache_is_opt_in_and_fresh_hashes_use_file_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = legacy_rust.resolve_legacy_arrow_manifest(_write_legacy_manifest(tmp_path))
    signatures_path = artifacts.source_paths["signatures"]
    calls: list[Path] = []
    real_digest_path = legacy_rust._digest_path

    def counting_digest_path(path: Path) -> legacy_rust.ArtifactDigest:
        calls.append(path)
        return real_digest_path(path)

    monkeypatch.setattr(legacy_rust, "_digest_path", counting_digest_path)
    first = legacy_rust.digest_legacy_artifacts(artifacts)
    legacy_rust.digest_legacy_artifacts(artifacts)
    assert calls.count(signatures_path) == 2

    original_stat = signatures_path.stat()
    signatures_path.write_bytes(b"SIGNATURES")
    os.utime(signatures_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = legacy_rust.digest_legacy_artifacts(artifacts)

    assert signatures_path.stat().st_size == original_stat.st_size
    assert signatures_path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert first["signatures"].sha256 != second["signatures"].sha256


def test_resolve_legacy_manifest_refuses_canonical_v2_dataset_manifest(tmp_path: Path) -> None:
    manifest_path = _write_legacy_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_provenance"] = {"normalization_version": "canonical_v2"}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="PRACTICE-ONLY.*refuses canonical_v2"):
        legacy_rust.resolve_legacy_arrow_manifest(manifest_path)


def test_resolve_legacy_manifest_refuses_canonical_v2_referenced_directory(tmp_path: Path) -> None:
    manifest_path = _write_legacy_manifest(tmp_path, child_normalization="canonical_v2")

    with pytest.raises(ValueError, match="PRACTICE-ONLY.*refuses canonical_v2"):
        legacy_rust.resolve_legacy_arrow_manifest(manifest_path)


def test_build_legacy_rust_featurizer_forces_preprocess_false_and_n_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = legacy_rust.resolve_legacy_arrow_manifest(_write_legacy_manifest(tmp_path))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    class FakeRustFeaturizer:
        @staticmethod
        def from_arrow_paths(*args: Any, **kwargs: Any) -> object:
            calls.append((args, kwargs))
            return sentinel

    monkeypatch.setattr(
        legacy_rust,
        "_load_s2and_rust",
        lambda: SimpleNamespace(RustFeaturizer=FakeRustFeaturizer),
    )

    result = legacy_rust.build_legacy_rust_featurizer(
        artifacts,
        n_jobs=20,
        signature_ids=(value for value in (2, "1")),
    )

    assert result is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0]["specter"] == str((tmp_path / "specter2.arrow").resolve())
    assert kwargs == {
        "signature_ids": ["2", "1"],
        "name_tuples": "filtered",
        "preprocess": False,
        "num_threads": 20,
    }


def test_build_legacy_rust_featurizer_rejects_duplicate_requested_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = legacy_rust.resolve_legacy_arrow_manifest(_write_legacy_manifest(tmp_path))

    def fail_load() -> Any:
        raise AssertionError("Rust must not be loaded")

    monkeypatch.setattr(legacy_rust, "_load_s2and_rust", fail_load)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        legacy_rust.build_legacy_rust_featurizer(artifacts, n_jobs=20, signature_ids=[1, "1"])


def test_build_rechecks_manifest_and_refuses_canonical_v2_that_lands_after_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_legacy_manifest(tmp_path)
    artifacts = legacy_rust.resolve_legacy_arrow_manifest(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["normalization_version"] = "canonical_v2"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_load() -> Any:
        raise AssertionError("Rust must not be loaded")

    monkeypatch.setattr(legacy_rust, "_load_s2and_rust", fail_load)

    with pytest.raises(ValueError, match="PRACTICE-ONLY.*refuses canonical_v2"):
        legacy_rust.build_legacy_rust_featurizer(artifacts, n_jobs=20)


@pytest.mark.parametrize("label", ["1", None, math.nan, math.inf, -1, 0.5, 2])
def test_validate_labeled_pairs_rejects_non_binary_labels(label: Any) -> None:
    expected_exception = TypeError if label is None or isinstance(label, str) else ValueError
    with pytest.raises(expected_exception, match="label"):
        legacy_rust.validate_labeled_pairs([("left", "right", label)])


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ((None, "right", 1), "left_id"),
        ((True, "right", 1), "left_id"),
        ((" ", "right", 1), "left_id"),
        (("same", "same", 1), "self-pair"),
        (("left", "right"), "exactly"),
    ],
)
def test_validate_labeled_pairs_rejects_invalid_ids_and_rows(pair: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        legacy_rust.validate_labeled_pairs([pair])


def test_signature_ids_for_labeled_pairs_preserves_first_seen_order() -> None:
    pairs = [(2, "1", True), ("3", "2", False)]

    assert legacy_rust.signature_ids_for_labeled_pairs(pairs) == ("2", "1", "3")


class _FakeMatrixFeaturizer:
    def __init__(self, signature_ids: list[str], *, malformed: bool = False) -> None:
        self._signature_ids = signature_ids
        self.malformed = malformed
        self.calls: list[tuple[list[tuple[int, int]], list[int], int, float]] = []

    def signature_ids(self) -> list[str]:
        return self._signature_ids

    def featurize_pairs_matrix_indexed(
        self,
        pairs: list[tuple[int, int]],
        selected_indices: list[int],
        n_jobs: int,
        nan_value: float,
    ) -> list[list[float]]:
        self.calls.append((pairs, selected_indices, n_jobs, nan_value))
        if self.malformed:
            return [[0.0]]
        return [
            [float(left * 100 + right * 10 + feature_index) for feature_index in selected_indices]
            for left, right in pairs
        ]


def test_featurize_labeled_pairs_indexes_ids_once_and_projects_main_and_nameless() -> None:
    featurizer = _FakeMatrixFeaturizer(["sig-b", "sig-a", "sig-c"])
    main_info = SimpleNamespace(features_to_use=["main"], feature_group_to_index={"main": [2, 0, 2]})
    nameless_info = SimpleNamespace(
        features_to_use=["nameless"],
        feature_group_to_index={"nameless": [2, 1]},
    )

    result = legacy_rust.featurize_labeled_pairs(
        featurizer,
        [("sig-a", "sig-b", 1), ("sig-c", "sig-a", 0)],
        featurization_info=main_info,
        nameless_featurization_info=nameless_info,
        n_jobs=20,
        nan_value=-999.0,
    )

    assert result.indexed_pairs == ((1, 0), (2, 1))
    assert result.signature_ids == ("sig-b", "sig-a", "sig-c")
    assert result.main_feature_indices == (0, 2)
    assert result.nameless_feature_indices == (1, 2)
    np.testing.assert_array_equal(result.labels, np.asarray([1.0, 0.0]))
    np.testing.assert_array_equal(result.main, np.asarray([[100.0, 102.0], [210.0, 212.0]]))
    assert result.nameless is not None
    np.testing.assert_array_equal(result.nameless, np.asarray([[101.0, 102.0], [211.0, 212.0]]))
    assert featurizer.calls == [([(1, 0), (2, 1)], [0, 1, 2], 20, -999.0)]
    main, labels, nameless = result.as_training_tuple()
    assert main is result.main
    assert labels is result.labels
    assert nameless is result.nameless


def test_featurize_labeled_pairs_rejects_missing_id_before_calling_rust() -> None:
    featurizer = _FakeMatrixFeaturizer(["present"])
    info = SimpleNamespace(features_to_use=[], feature_group_to_index={})

    with pytest.raises(KeyError, match="absent.*missing"):
        legacy_rust.featurize_labeled_pairs(
            featurizer,
            [("present", "missing", 1)],
            featurization_info=info,
            nameless_featurization_info=None,
            n_jobs=20,
        )

    assert featurizer.calls == []


def test_featurize_labeled_pairs_returns_typed_empty_shapes_without_calling_rust() -> None:
    featurizer = _FakeMatrixFeaturizer(["present"])
    main_info = SimpleNamespace(features_to_use=["main"], feature_group_to_index={"main": [2, 0]})
    nameless_info = SimpleNamespace(features_to_use=[], feature_group_to_index={})

    result = legacy_rust.featurize_labeled_pairs(
        featurizer,
        [],
        featurization_info=main_info,
        nameless_featurization_info=nameless_info,
        n_jobs=20,
    )

    assert result.main.shape == (0, 2)
    assert result.labels.shape == (0,)
    assert result.nameless is not None
    assert result.nameless.shape == (0, 0)
    assert featurizer.calls == []


def test_featurize_labeled_pairs_rejects_malformed_native_matrix() -> None:
    featurizer = _FakeMatrixFeaturizer(["left", "right"], malformed=True)
    info = SimpleNamespace(features_to_use=["main"], feature_group_to_index={"main": [0, 1]})

    with pytest.raises(ValueError, match=r"shape \(1, 1\).*expected \(1, 2\)"):
        legacy_rust.featurize_labeled_pairs(
            featurizer,
            [("left", "right", 1)],
            featurization_info=info,
            nameless_featurization_info=None,
            n_jobs=20,
        )
