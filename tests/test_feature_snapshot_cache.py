"""Tests for the training-time featurized-split snapshot cache."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.feature_cache as feature_cache_mod
from s2and.data import ANDData
from s2and.feature_cache import build_and_cached_featurize
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.name_counts_index import NameCountsIndex
from s2and.text import compute_block
from tests.helpers import tiny_name_counts_index


def _build_dataset(
    n_jobs: int = 1,
    *,
    signatures: str = "tests/dummy/signatures.json",
    papers: str = "tests/dummy/papers.json",
    clusters: str = "tests/dummy/clusters.json",
    name: str = "snapshot_cache_dummy",
    name_counts_index: NameCountsIndex | None = None,
    name_tuples: set[tuple[str, str]] | None = None,
    preprocess: bool = True,
    compute_block_fn: Callable[[str], str] = compute_block,
    capture_feature_sources: bool = True,
) -> ANDData:
    return ANDData(
        **_dataset_kwargs(
            n_jobs,
            signatures=signatures,
            papers=papers,
            clusters=clusters,
            name=name,
            name_counts_index=name_counts_index,
            name_tuples=name_tuples,
            preprocess=preprocess,
            compute_block_fn=compute_block_fn,
        ),
        _capture_feature_source_hashes=capture_feature_sources,
    )


def _dataset_kwargs(
    n_jobs: int = 1,
    *,
    signatures: str = "tests/dummy/signatures.json",
    papers: str = "tests/dummy/papers.json",
    clusters: str = "tests/dummy/clusters.json",
    name: str = "snapshot_cache_dummy",
    name_counts_index: NameCountsIndex | None = None,
    name_tuples: set[tuple[str, str]] | None = None,
    preprocess: bool = True,
    compute_block_fn: Callable[[str], str] = compute_block,
) -> dict[str, Any]:
    return {
        "signatures": signatures,
        "papers": papers,
        "clusters": clusters,
        "name": name,
        "mode": "train",
        "name_counts_index": name_counts_index,
        "name_tuples": set() if name_tuples is None else name_tuples,
        "preprocess": preprocess,
        "compute_block_fn": compute_block_fn,
        "n_jobs": n_jobs,
        "train_pairs_size": 100,
        "val_pairs_size": 50,
        "test_pairs_size": 50,
    }


def _synthetic_split_pairs(dataset: ANDData):
    cluster_of = dataset.signature_to_cluster_id or {}
    all_pairs = [
        (str(a), str(b), float(cluster_of.get(a) == cluster_of.get(b)))
        for a, b in combinations(sorted(dataset.signatures), 2)
    ]
    return all_pairs[0::3], all_pairs[1::3], all_pairs[2::3]


@pytest.fixture()
def dummy_builder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    # The 9-signature dummy dataset is below the real splitter's minimum block
    # count, so pin deterministic synthetic splits for these tests.
    monkeypatch.setattr(feature_cache_mod, "resolve_training_pairs", _synthetic_split_pairs)
    return _dataset_kwargs()


def test_snapshot_hit_skips_recompute_and_is_bit_identical(
    dummy_builder: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    _, first = build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path, n_jobs=1)

    def fail_if_computed(*_args, **_kwargs):
        raise AssertionError("second run must load snapshots, not recompute")

    monkeypatch.setattr(feature_cache_mod, "many_pairs_featurize", fail_if_computed)
    _, second = build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path, n_jobs=1)

    for (features_1, labels_1, nameless_1), (features_2, labels_2, nameless_2) in zip(first, second, strict=True):
        assert features_1.tobytes() == features_2.tobytes()
        assert labels_1.tobytes() == labels_2.tobytes()
        assert nameless_1 is None and nameless_2 is None


def test_caller_cannot_control_source_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    anddata_kwargs = _dataset_kwargs()
    anddata_kwargs["_capture_feature_source_hashes"] = False

    with pytest.raises(ValueError, match="source capture is owned"):
        build_and_cached_featurize(
            anddata_kwargs,
            FeaturizationInfo(features_to_use=["year_diff"]),
            cache_dir=tmp_path,
        )


def test_arrow_inputs_are_outside_python_snapshot_boundary(tmp_path: Path) -> None:
    anddata_kwargs = _dataset_kwargs()
    anddata_kwargs["_validated_arrow_inputs"] = object()

    with pytest.raises(ValueError, match="only classic file-backed Python"):
        build_and_cached_featurize(
            anddata_kwargs,
            FeaturizationInfo(features_to_use=["year_diff"]),
            cache_dir=tmp_path,
        )


def test_captured_provenance_is_consumed_once(dummy_builder: dict[str, Any], tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    dataset, _ = build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path)
    paper_id = str(dataset.signatures["0"].paper_id)
    dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=1950)

    with pytest.raises(ValueError, match="freshly constructed ANDData"):
        dataset._consume_feature_source_sha256()


def test_source_capture_does_not_use_path_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(feature_cache_mod, "resolve_training_pairs", _synthetic_split_pairs)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("source capture must stream bytes while parsing")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    _, splits = build_and_cached_featurize(
        _dataset_kwargs(),
        FeaturizationInfo(features_to_use=["year_diff"]),
        cache_dir=tmp_path,
    )
    assert sum(split[0].shape[0] for split in splits) == 36


def test_specter_capture_streams_and_hashes_the_full_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    base_dir = _copy_dummy_files(tmp_path)
    probe = _build_file_backed_dataset(base_dir)
    specter_path = base_dir / "specter.pkl"
    with open(specter_path, "wb") as handle:
        pickle.dump(
            {str(paper_id): np.ones(4, dtype=np.float64) for paper_id in probe.papers},
            handle,
        )
        handle.write(b"trailing bytes are part of the source identity")
    with open(specter_path, "rb") as handle:
        expected_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("SPECTER capture must not buffer the whole pickle")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    dataset = ANDData(
        str(base_dir / "signatures.json"),
        str(base_dir / "papers.json"),
        clusters=str(base_dir / "clusters.json"),
        name="snapshot_cache_specter_stream",
        specter_embeddings=str(specter_path),
        name_counts_index=None,
        name_tuples=set(),
        preprocess=True,
        n_jobs=1,
        _capture_feature_source_hashes=True,
    )
    assert dataset._consume_feature_source_sha256()["specter_embeddings"] == expected_sha256


@pytest.mark.parametrize(
    ("featurizer_info", "nameless_featurizer_info"),
    [
        (FeaturizationInfo(features_to_use=["name_counts"]), None),
        (
            FeaturizationInfo(features_to_use=["year_diff"]),
            FeaturizationInfo(features_to_use=["name_counts"]),
        ),
    ],
)
def test_name_counts_selection_requires_verified_provenance(
    dummy_builder: dict[str, Any],
    tmp_path: Path,
    featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo | None,
) -> None:
    with pytest.raises(ValueError, match="verified name-count provenance"):
        build_and_cached_featurize(
            dummy_builder,
            featurizer_info,
            cache_dir=tmp_path,
            nameless_featurizer_info=nameless_featurizer_info,
        )


def test_name_counts_provenance_is_not_required_when_feature_is_not_selected(
    dummy_builder: dict[str, Any], tmp_path: Path
) -> None:
    _, splits = build_and_cached_featurize(
        dummy_builder,
        FeaturizationInfo(features_to_use=["year_diff"]),
        cache_dir=tmp_path,
    )
    assert len(splits) == 3


def test_unused_name_counts_provenance_does_not_change_snapshot_identity(
    dummy_builder: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path)

    def fail_if_computed(*_args, **_kwargs):
        raise AssertionError("unused name-count provenance must not cause a miss")

    monkeypatch.setattr(feature_cache_mod, "many_pairs_featurize", fail_if_computed)
    with_name_counts = dict(dummy_builder)
    with_name_counts["name_counts_index"] = tiny_name_counts_index()
    build_and_cached_featurize(with_name_counts, featurizer_info, cache_dir=tmp_path)


def test_name_counts_selection_accepts_verified_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(feature_cache_mod, "resolve_training_pairs", _synthetic_split_pairs)

    _, splits = build_and_cached_featurize(
        _dataset_kwargs(name_counts_index=tiny_name_counts_index()),
        FeaturizationInfo(features_to_use=["name_counts"]),
        cache_dir=tmp_path,
    )
    assert len(splits) == 3


def test_delimiter_colliding_name_tuples_have_distinct_fingerprints() -> None:
    one_pair = _build_dataset(name_tuples={("a", "b\nc\td")})
    two_pairs = _build_dataset(name_tuples={("a", "b"), ("c", "d")})
    assert "\n".join(f"{first}\t{second}" for first, second in sorted(one_pair.name_tuples)) == "\n".join(
        f"{first}\t{second}" for first, second in sorted(two_pairs.name_tuples)
    )

    one_fingerprint = feature_cache_mod._dataset_feature_fingerprint(
        one_pair,
        one_pair._consume_feature_source_sha256(),
        uses_name_counts=False,
    )
    two_fingerprint = feature_cache_mod._dataset_feature_fingerprint(
        two_pairs,
        two_pairs._consume_feature_source_sha256(),
        uses_name_counts=False,
    )
    assert one_fingerprint["name_tuples_sha256"] != two_fingerprint["name_tuples_sha256"]


@pytest.mark.parametrize("invalid_digest", [None, "0" * 63, "A" * 64, "g" * 64])
def test_fingerprint_rejects_malformed_source_digests(invalid_digest: object) -> None:
    dataset = _build_dataset()
    source_hashes = dataset._consume_feature_source_sha256()
    source_hashes["signatures"] = invalid_digest  # type: ignore[assignment]

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        feature_cache_mod._dataset_feature_fingerprint(dataset, source_hashes, uses_name_counts=False)


def test_snapshot_key_changes_with_featurizer_version_and_pairs(
    dummy_builder: dict[str, Any],
) -> None:
    dataset = _build_dataset()
    fingerprint = feature_cache_mod._dataset_feature_fingerprint(
        dataset,
        dataset._consume_feature_source_sha256(),
        uses_name_counts=False,
    )

    def key(*, version: int = 10, label: float = 1.0) -> str:
        return feature_cache_mod._snapshot_key(
            dataset_fingerprint=fingerprint,
            featurizer_info=FeaturizationInfo(features_to_use=["year_diff"], featurizer_version=version),
            nameless_featurizer_info=None,
            nan_value=np.nan,
            pair_list_hash=feature_cache_mod._hash_pair_list([("a", "b", label)]),
        )

    base_key = key()
    assert key(version=11) != base_key
    assert key(label=0.0) != base_key
    assert key() == base_key


def test_cold_arrays_are_validated_before_publication(
    dummy_builder: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_featurize = feature_cache_mod.many_pairs_featurize
    original_publish = feature_cache_mod._publish_snapshot

    def invalid_featurize(pairs, *_args, **_kwargs):
        return (
            np.zeros((len(pairs), 1), dtype=np.float32),
            np.zeros(len(pairs), dtype=np.float64),
            None,
        )

    def fail_publish(*_args, **_kwargs):
        raise AssertionError("invalid arrays must not reach publication")

    monkeypatch.setattr(feature_cache_mod, "many_pairs_featurize", invalid_featurize)
    monkeypatch.setattr(feature_cache_mod, "_publish_snapshot", fail_publish)

    with pytest.raises(ValueError, match="dtype float32"):
        build_and_cached_featurize(
            dummy_builder,
            FeaturizationInfo(features_to_use=["year_diff"]),
            cache_dir=tmp_path,
        )
    assert list(tmp_path.glob("*.npz")) == []

    # Validation failure releases the build claim so a bounded retry can own
    # and publish the same content-addressed snapshots.
    monkeypatch.setattr(feature_cache_mod, "many_pairs_featurize", original_featurize)
    monkeypatch.setattr(feature_cache_mod, "_publish_snapshot", original_publish)
    _, retried = build_and_cached_featurize(
        dummy_builder,
        FeaturizationInfo(features_to_use=["year_diff"]),
        cache_dir=tmp_path,
    )
    assert len(retried) == 3
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_corrupt_snapshots_raise_contextual_errors(dummy_builder: dict[str, Any], tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    for corruption in ("invalid", "truncated"):
        cache_dir = tmp_path / corruption
        build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=cache_dir)
        snapshot = sorted(cache_dir.glob("*.npz"))[0]
        payload = snapshot.read_bytes()
        snapshot.write_bytes(b"not an npz file" if corruption == "invalid" else payload[: len(payload) // 2])

        with pytest.raises(ValueError, match="unreadable"):
            build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=cache_dir)


def test_snapshot_publication_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.npz"
    first = feature_cache_mod._publish_snapshot(path, {"X": np.zeros((1, 1)), "y": np.zeros(1)})
    original_bytes = path.read_bytes()
    second = feature_cache_mod._publish_snapshot(path, {"X": np.ones((2, 2)), "y": np.ones(2)})
    assert first is True
    assert second is False
    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob("*.tmp")) == []


def test_nameless_snapshot_members_round_trip(
    dummy_builder: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    nameless_info = FeaturizationInfo(features_to_use=["year_diff"])
    _, first = build_and_cached_featurize(
        dummy_builder,
        featurizer_info,
        cache_dir=tmp_path,
        nameless_featurizer_info=nameless_info,
    )

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("expected snapshot hits")

    monkeypatch.setattr(
        feature_cache_mod,
        "many_pairs_featurize",
        fail_if_recomputed,
    )
    _, second = build_and_cached_featurize(
        dummy_builder,
        featurizer_info,
        cache_dir=tmp_path,
        nameless_featurizer_info=nameless_info,
    )
    for (_, _, nameless_1), (_, _, nameless_2) in zip(first, second, strict=True):
        assert nameless_1 is not None and nameless_2 is not None
        assert nameless_1.tobytes() == nameless_2.tobytes()


def _copy_dummy_files(tmp_path: Path) -> Path:
    base_dir = tmp_path / "dataset_files"
    base_dir.mkdir()
    for filename in ("signatures.json", "papers.json", "clusters.json"):
        shutil.copy(Path("tests/dummy") / filename, base_dir / filename)
    return base_dir


def _build_file_backed_dataset(
    base_dir: Path,
    *,
    name: str = "snapshot_cache_files",
    specter_embeddings: dict[str, np.ndarray] | None = None,
    preprocess: bool = True,
    compute_block_fn: Callable[[str], str] = compute_block,
) -> ANDData:
    return ANDData(
        **_file_backed_kwargs(
            base_dir,
            name=name,
            specter_embeddings=specter_embeddings,
            preprocess=preprocess,
            compute_block_fn=compute_block_fn,
        ),
        _capture_feature_source_hashes=True,
    )


def _file_backed_kwargs(
    base_dir: Path,
    *,
    name: str = "snapshot_cache_files",
    specter_embeddings: dict[str, np.ndarray] | None = None,
    preprocess: bool = True,
    compute_block_fn: Callable[[str], str] = compute_block,
) -> dict[str, Any]:
    return {
        "signatures": str(base_dir / "signatures.json"),
        "papers": str(base_dir / "papers.json"),
        "clusters": str(base_dir / "clusters.json"),
        "name": name,
        "mode": "train",
        "specter_embeddings": specter_embeddings,
        "name_counts_index": None,
        "name_tuples": set(),
        "preprocess": preprocess,
        "compute_block_fn": compute_block_fn,
        "n_jobs": 1,
    }


def test_changed_file_bytes_produce_new_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(feature_cache_mod, "resolve_training_pairs", _synthetic_split_pairs)
    base_dir = _copy_dummy_files(tmp_path)
    cache_dir = tmp_path / "cache"
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])

    anddata_kwargs = _file_backed_kwargs(base_dir)
    build_and_cached_featurize(anddata_kwargs, featurizer_info, cache_dir=cache_dir)
    before_names = {path.name for path in cache_dir.glob("*.npz")}
    assert len(before_names) == 3

    papers_path = base_dir / "papers.json"
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    first_paper = next(iter(papers.values()))
    first_paper["year"] = (first_paper.get("year") or 2000) + 1
    papers_path.write_text(json.dumps(papers), encoding="utf-8")

    build_and_cached_featurize(anddata_kwargs, featurizer_info, cache_dir=cache_dir)
    after_names = {path.name for path in cache_dir.glob("*.npz")}
    assert len(after_names) == 6
    assert before_names < after_names


def test_fingerprint_rejects_in_memory_specter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    base_dir = _copy_dummy_files(tmp_path)
    probe = _build_file_backed_dataset(base_dir)
    paper_id = str(next(iter(probe.papers)))

    with pytest.raises(ValueError, match="in-memory embeddings"):
        build_and_cached_featurize(
            _file_backed_kwargs(
                base_dir,
                name="snapshot_cache_mem_specter",
                specter_embeddings={paper_id: np.ones(4, dtype=np.float64)},
            ),
            FeaturizationInfo(features_to_use=["year_diff"]),
            cache_dir=tmp_path / "cache",
        )


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda base_dir: _file_backed_kwargs(base_dir, preprocess=False), "preprocess=True"),
        (
            lambda base_dir: _file_backed_kwargs(base_dir, compute_block_fn=lambda name: name),
            "compute_block",
        ),
    ],
)
def test_fingerprint_rejects_unsupported_configurations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[Path], dict[str, Any]],
    message: str,
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    base_dir = _copy_dummy_files(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_and_cached_featurize(
            builder(base_dir),
            FeaturizationInfo(features_to_use=["year_diff"]),
            cache_dir=tmp_path / "cache",
        )


def test_malformed_readable_npz_and_key_mismatch_rejected(dummy_builder: dict[str, Any], tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path)
    snapshots = sorted(tmp_path.glob("*.npz"))
    snapshot = snapshots[0]

    np.savez_compressed(snapshot, wrong=np.zeros(3))
    with pytest.raises(ValueError, match="has members"):
        build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path)

    with np.load(snapshots[1], allow_pickle=False) as valid:
        arrays = {name: valid[name] for name in valid.files}
    arrays["key"] = np.array("0" * 64)
    np.savez_compressed(snapshot, **arrays)
    with pytest.raises(ValueError, match="embedded key"):
        build_and_cached_featurize(dummy_builder, featurizer_info, cache_dir=tmp_path)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        (
            {"key": np.array("k" * 64), "X": np.zeros((2, 2)), "y": np.zeros(2)},
            "expected \\(rows, 1\\)",
        ),
        (
            {"key": np.array("k" * 64), "X": np.zeros((2, 1)), "y": np.zeros(1)},
            "expected \\(2,\\)",
        ),
        (
            {
                "key": np.array("k" * 64),
                "X": np.zeros((2, 1)),
                "y": np.zeros(2),
                "nameless_X": np.zeros((1, 1)),
            },
            "nameless_X has shape",
        ),
    ],
)
def test_snapshot_shape_validation(arrays: dict[str, np.ndarray], message: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=message):
        feature_cache_mod._validate_snapshot_arrays(
            arrays,
            path=tmp_path / "shape.npz",
            expected_key="k" * 64,
            expected_rows=2,
            expected_width=1,
            expected_nameless_width=1 if "nameless_X" in arrays else None,
        )


def test_snapshot_permission_error_is_not_mislabeled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(feature_cache_mod.np, "load", denied)
    with pytest.raises(PermissionError, match="denied"):
        feature_cache_mod._load_snapshot(
            tmp_path / "unreadable.npz",
            expected_key="k" * 64,
            expected_rows=2,
            expected_width=1,
            expected_nameless_width=None,
        )


def test_failed_publication_cleans_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_savez(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(feature_cache_mod.np, "savez", broken_savez)
    path = tmp_path / "snapshot.npz"
    with pytest.raises(OSError, match="disk full"):
        feature_cache_mod._publish_snapshot(path, {"X": np.zeros((1, 1)), "y": np.zeros(1)})
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def _competing_publish_worker(path_str: str) -> bool:
    """Top-level worker for the multiprocessing write-once test."""
    from s2and.feature_cache import _publish_snapshot

    arrays = {
        "key": np.array("k" * 64),
        "X": np.arange(12, dtype=np.float64).reshape(3, 4),
        "y": np.zeros(3, dtype=np.float64),
    }
    return _publish_snapshot(Path(path_str), arrays)


def test_concurrent_competing_writers_publish_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "contended.npz"
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_competing_publish_worker, [str(path)] * 8))
    assert results.count(True) == 1
    assert results.count(False) == 7

    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {"key", "X", "y"}
        np.testing.assert_array_equal(loaded["X"], np.arange(12, dtype=np.float64).reshape(3, 4))
    assert list(tmp_path.glob("*.tmp")) == []


def _contended_cache_build_worker(cache_dir_str: str, counter_path_str: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Build the same three cold snapshots and record each real computation."""

    import os

    from s2and._atomic_io import exclusive_file_lock

    os.environ["S2AND_BACKEND"] = "python"
    cache_dir = Path(cache_dir_str)
    counter_path = Path(counter_path_str)
    dummy_dir = Path("tests/dummy").resolve()

    def record(event: str) -> None:
        with exclusive_file_lock(f"{counter_path}.lock"):
            with counter_path.open("a", encoding="utf-8") as counter:
                counter.write(f"{event}\n")

    record("ready")
    ready_deadline = time.monotonic() + 10.0
    while True:
        with exclusive_file_lock(f"{counter_path}.lock"):
            events = counter_path.read_text(encoding="utf-8").splitlines()
        if events.count("ready") == 2:
            break
        if time.monotonic() >= ready_deadline:
            raise TimeoutError("timed out waiting for both feature-cache test workers")
        time.sleep(0.01)

    def fixed_pairs(_dataset):
        pairs = [("0", "1", 1.0)]
        return pairs, pairs, pairs

    def recording_featurize(pairs, _dataset, info, **_kwargs):
        record("compute")
        time.sleep(0.2)
        width = len(info.selected_feature_indices())
        return (
            np.zeros((len(pairs), width), dtype=np.float64),
            np.zeros(len(pairs), dtype=np.float64),
            None,
        )

    original_savez = feature_cache_mod.np.savez

    def recording_savez(*args, **kwargs):
        record("serialize")
        return original_savez(*args, **kwargs)

    feature_cache_mod.resolve_training_pairs = fixed_pairs
    feature_cache_mod.many_pairs_featurize = recording_featurize
    feature_cache_mod.np.savez = recording_savez
    kwargs = _dataset_kwargs(
        signatures=str(dummy_dir / "signatures.json"),
        papers=str(dummy_dir / "papers.json"),
        clusters=str(dummy_dir / "clusters.json"),
    )
    _, splits = build_and_cached_featurize(
        kwargs,
        FeaturizationInfo(features_to_use=["year_diff"]),
        cache_dir=cache_dir,
        n_jobs=1,
    )
    return os.getpid(), tuple(features.shape for features, _labels, _nameless in splits)


def test_concurrent_cold_builds_compute_each_snapshot_once(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    counter_path = tmp_path / "compute-count.txt"
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                _contended_cache_build_worker,
                [str(cache_dir)] * 2,
                [str(counter_path)] * 2,
            )
        )

    assert len({worker_pid for worker_pid, _shapes in results}) == 2
    assert [shapes for _worker_pid, shapes in results] == [((1, 1), (1, 1), (1, 1))] * 2
    events = counter_path.read_text(encoding="utf-8").splitlines()
    assert events.count("ready") == 2
    assert events.count("compute") == 3
    assert events.count("serialize") == 3
    assert len(list(cache_dir.glob("*.npz"))) == 3


def test_end_to_end_cached_matches_uncached_with_real_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed-pairs cached output is bit-identical to uncached output."""
    monkeypatch.setenv("S2AND_BACKEND", "python")
    base_dir = _copy_dummy_files(tmp_path)
    pair_rows = {
        "train": ["0,1,YES", "0,2,NO", "3,4,YES", "5,6,NO"],
        "val": ["1,2,NO", "4,5,YES"],
        "test": ["7,8,NO", "6,7,YES"],
    }
    for split_name, rows in pair_rows.items():
        (base_dir / f"{split_name}_pairs.csv").write_text(
            "\n".join(["pair1,pair2,label", *rows]) + "\n", encoding="utf-8"
        )

    anddata_kwargs: dict[str, Any] = {
        "signatures": str(base_dir / "signatures.json"),
        "papers": str(base_dir / "papers.json"),
        "name": "snapshot_cache_fixed_pairs",
        "mode": "train",
        "train_pairs": str(base_dir / "train_pairs.csv"),
        "val_pairs": str(base_dir / "val_pairs.csv"),
        "test_pairs": str(base_dir / "test_pairs.csv"),
        "name_counts_index": None,
        "name_tuples": set(),
        "preprocess": True,
        "n_jobs": 1,
    }

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    nameless_info = FeaturizationInfo(features_to_use=["year_diff"])
    uncached = featurize(
        ANDData(**anddata_kwargs),
        featurizer_info,
        n_jobs=1,
        nameless_featurizer_info=nameless_info,
    )

    cache_dir = tmp_path / "cache"

    _, cold = build_and_cached_featurize(
        anddata_kwargs,
        featurizer_info,
        cache_dir=cache_dir,
        nameless_featurizer_info=nameless_info,
    )
    _, warm = build_and_cached_featurize(
        anddata_kwargs,
        featurizer_info,
        cache_dir=cache_dir,
        nameless_featurizer_info=nameless_info,
    )
    assert len(list(cache_dir.glob("*.npz"))) == 3

    for uncached_split, cold_split, warm_split in zip(uncached, cold, warm, strict=True):
        for uncached_array, cold_array, warm_array in zip(uncached_split, cold_split, warm_split, strict=True):
            assert uncached_array is not None
            assert cold_array is not None
            assert warm_array is not None
            assert uncached_array.tobytes() == cold_array.tobytes() == warm_array.tobytes()
