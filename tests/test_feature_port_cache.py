import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import s2and.feature_port as feature_port
import s2and.runtime as runtime
from s2and.arrow_inputs import validate_arrow_training_artifacts
from s2and.consts import NORMALIZATION_VERSION
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple, write_test_arrow_artifact_manifest


class DummyRustFeaturizer:
    created = []
    signature_overlay_payloads = []

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def signature_ids(self):
        return []

    def get_constraints_matrix_indexed(self, *_args, **_kwargs):
        return []

    @classmethod
    def from_arrow_paths(cls, *_args, **_kwargs):
        raise AssertionError("cache tests route builds through the patched feature_port.build_rust_featurizer")

    def update_signature_name_counts(self, signatures):
        self.__class__.signature_overlay_payloads.append(signatures)
        return len(signatures)

    def featurize_pairs_matrix_indexed(self, *_args, **_kwargs):
        return []

    @classmethod
    def load(cls, _path):
        raise AssertionError("Disk cache path should not be used in this test")

    def update_cluster_seeds(self, _require_map, _disallow_set):
        self.cluster_seeds_require_state = {str(key): str(value) for key, value in dict(_require_map).items()}
        self.cluster_seeds_disallow_state = {(str(left), str(right)) for left, right in set(_disallow_set)}

    def cluster_seeds_require(self):
        return list(getattr(self, "cluster_seeds_require_state", {}).items())

    def cluster_seeds_disallow(self):
        return list(getattr(self, "cluster_seeds_disallow_state", set()))


class DummyRustModule:
    __version__ = runtime.REQUIRED_RUST_EXTENSION_VERSION
    RustFeaturizer = DummyRustFeaturizer


class DummyDataset(ANDData):
    def __init__(self, name: str, mode: str = "train"):
        self.name = name
        self.mode = mode
        self.signatures = {}
        self.papers = {}
        self.name_tuples = set()
        self.preprocess = True
        self.n_jobs = 1
        self.original_signatures_path = None
        self.original_papers_path = None
        self.cluster_seeds_require = {}
        self.cluster_seeds_disallow = set()
        self.signatures_path = None
        self.papers_path = None
        self.clusters_path = None
        self.cluster_seeds_path = None
        self.specter_embeddings_path = None
        self.arrow_paths = {"signatures": f"{name}.arrow"}
        self.arrow_artifact_generation = f"test-generation-{name}"
        self.name_counts_provenance = tiny_name_counts_provenance()


def _cache_size() -> int:
    return sum(len(entries) for entries in feature_port._RUST_FEATURIZER_CACHE.values())


def _cache_keys(dataset: DummyDataset) -> list[feature_port._RustFeaturizerCacheKey]:
    return list(feature_port._RUST_FEATURIZER_CACHE[dataset])


def _dummy_build_rust_featurizer(dataset):
    """Dataset-aware stand-in for the Arrow featurizer build door.

    The cache machinery under test only cares that a build happened for a
    given dataset/cache-key; the real build (from_arrow_paths over validated
    Arrow artifacts) is covered by tests/test_arrow_training_ingestion.py.
    """

    DummyRustFeaturizer.created.append(dataset.name)
    return (
        DummyRustFeaturizer(dataset.name),
        {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
    )


@pytest.fixture(autouse=True)
def _reset_feature_port_state(monkeypatch):
    feature_port.clear_rust_featurizer_cache()
    DummyRustFeaturizer.created = []
    DummyRustFeaturizer.signature_overlay_payloads = []
    monkeypatch.setattr(feature_port, "s2and_rust", DummyRustModule)
    monkeypatch.setattr(feature_port, "build_rust_featurizer", _dummy_build_rust_featurizer)
    yield
    feature_port.clear_rust_featurizer_cache()


def test_rust_featurizer_in_memory_cache_keeps_train_entries():
    """Same-process cache keeps all live dataset entries."""
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    feature_port._get_rust_featurizer(d1)
    assert _cache_size() == 1

    feature_port._get_rust_featurizer(d2)
    assert _cache_size() == 2
    assert feature_port._RUST_FEATURIZER_CACHE.get(d1) is not None
    assert feature_port._RUST_FEATURIZER_CACHE.get(d2) is not None

    # Re-access d1 — should be a cache hit, no rebuild.
    feature_port._get_rust_featurizer(d1)
    assert DummyRustFeaturizer.created == ["d1", "d2"]


def test_rust_featurizer_cache_tracks_cluster_seed_version():
    dataset = DummyDataset("seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = 1
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == ["seed_version_cache_dataset", "seed_version_cache_dataset"]
    assert _cache_size() == 1
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


def test_rust_featurizer_cache_key_uses_retained_generation_and_exact_paths():
    dataset = DummyDataset("arrow_generation_cache_dataset", mode="train")
    first_key = feature_port._rust_featurizer_cache_key(dataset)
    dataset.arrow_artifact_generation = "replacement-generation"
    second_key = feature_port._rust_featurizer_cache_key(dataset)
    dataset.arrow_paths = {"signatures": "replacement.arrow"}
    third_key = feature_port._rust_featurizer_cache_key(dataset)

    assert second_key != first_key
    assert third_key != second_key


def test_rust_featurizer_cache_key_does_not_recheck_immutable_files(tmp_path: Path):
    dataset = DummyDataset("name_counts_index_metadata_cache_dataset", mode="train")
    index_dir = tmp_path / "name_counts_index"
    generation_dir = index_dir / "generations" / "gen-1"
    generation_dir.mkdir(parents=True)
    child_path = generation_dir / "first.bin"
    child_path.write_bytes(b"first")
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "test-fingerprint",
                "files": {"first": {"path": "generations/gen-1/first.bin"}},
            }
        ),
        encoding="utf-8",
    )
    dataset.arrow_paths = {"name_counts_index": str(index_dir)}

    first_key = feature_port._rust_featurizer_cache_key(dataset)
    child_path.write_bytes(b"second payload")
    second_key = feature_port._rust_featurizer_cache_key(dataset)

    assert second_key == first_key


def test_rust_featurizer_cache_key_rechecks_non_seed_fingerprint(monkeypatch):
    dataset = DummyDataset("memoized_non_seed_cache_dataset", mode="train")
    calls = {"count": 0}

    def fake_source_paths(_dataset):
        calls["count"] += 1
        return ()

    monkeypatch.setattr(feature_port, "_rust_featurizer_source_paths", fake_source_paths)

    first_key = feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=0)
    second_key = feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)

    assert calls["count"] == 2
    assert first_key.non_seed == second_key.non_seed
    assert first_key.seed.cluster_seeds_version == 0
    assert second_key.seed.cluster_seeds_version == 1


def test_rust_featurizer_cache_retries_when_seed_version_changes_during_lookup(monkeypatch):
    dataset = DummyDataset("seed_version_race_dataset", mode="train")
    versions = [0, 1]

    def next_seed_version(_dataset):
        if versions:
            return versions.pop(0)
        return 1

    monkeypatch.setattr(feature_port, "_cluster_seeds_version_for_cache", next_seed_version)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 3)

    featurizer = feature_port._get_rust_featurizer(dataset)

    assert featurizer.dataset_name == "seed_version_race_dataset"
    assert DummyRustFeaturizer.created == ["seed_version_race_dataset"]
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


def test_rust_featurizer_cache_retries_when_seed_version_changes_during_build(monkeypatch):
    dataset = DummyDataset("seed_version_build_race_dataset", mode="train")
    dataset._cluster_seeds_version = 0
    build_calls = {"count": 0}

    def _build_stub(dataset_arg):
        build_calls["count"] += 1
        if build_calls["count"] == 1:
            dataset_arg._cluster_seeds_version = 1
            featurizer_name = "stale"
        else:
            featurizer_name = "fresh"
        return (
            DummyRustFeaturizer(featurizer_name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)

    featurizer = feature_port._get_rust_featurizer(dataset)

    assert featurizer.dataset_name == "fresh"
    assert build_calls["count"] == 2
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


@pytest.mark.parametrize(
    ("case_name", "mutate_dataset"),
    [
        ("preprocess", lambda dataset: setattr(dataset, "preprocess", False)),
        ("n_jobs", lambda dataset: setattr(dataset, "n_jobs", 2)),
        ("name_tuples", lambda dataset: dataset.name_tuples.add(("bill", "william"))),
    ],
)
def test_rust_featurizer_cache_rebuilds_for_material_dataset_fields(case_name, mutate_dataset):
    dataset = DummyDataset(f"material_cache_{case_name}", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    mutate_dataset(dataset)
    second = feature_port._get_rust_featurizer(dataset)
    removed = feature_port.evict_rust_featurizer(dataset)
    third = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert removed is True
    assert third is not second
    assert DummyRustFeaturizer.created == [dataset.name, dataset.name, dataset.name]
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset)]


@pytest.mark.parametrize(
    "mutate_dataset",
    [
        lambda dataset: setattr(dataset, "signatures_path", "new_signatures.json"),
        lambda dataset: dataset.signatures.__setitem__("s1", object()),
        lambda dataset: dataset.papers.__setitem__("p1", object()),
        lambda dataset: setattr(dataset, "specter_embeddings", {"p1": object()}),
    ],
)
def test_rust_featurizer_cache_ignores_unconsumed_python_state(mutate_dataset):
    dataset = DummyDataset("unconsumed_python_state", mode="train")
    first = feature_port._get_rust_featurizer(dataset)
    mutate_dataset(dataset)
    second = feature_port._get_rust_featurizer(dataset)

    assert second is first
    assert DummyRustFeaturizer.created == [dataset.name]


def test_rust_featurizer_cache_rebuilds_for_in_place_numpy_embedding_mutation(monkeypatch):
    snapshots: list[float] = []

    def _embedding_build(dataset_arg):
        snapshots.append(float(dataset_arg.specter_embeddings["p"][0]))
        DummyRustFeaturizer.created.append(dataset_arg.name)
        return (
            DummyRustFeaturizer(dataset_arg.name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
        )

    monkeypatch.setattr(feature_port, "build_rust_featurizer", _embedding_build)
    dataset = DummyDataset("embedding_mutation_cache_dataset", mode="train")
    dataset.specter_embeddings = {"p": np.asarray([0.0], dtype=np.float32)}

    first = feature_port._get_rust_featurizer(dataset)
    dataset.specter_embeddings["p"][0] = 1.0
    second = feature_port._get_rust_featurizer(dataset)
    removed = feature_port.evict_rust_featurizer(dataset)
    third = feature_port._get_rust_featurizer(dataset)

    assert second is first
    assert removed is True
    assert third is not second
    assert snapshots == [0.0, 1.0]


def test_validated_arrow_generation_ignores_unconsumed_python_rows_and_rekeys_generation(tmp_path: Path) -> None:
    signatures_path = tmp_path / "immutable.arrow"

    def publish_generation(payload: bytes) -> str:
        signatures_path.write_bytes(payload)
        files = {
            "signatures": {
                "path": signatures_path.name,
                "kind": "file",
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        generation_id = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_generation": {
                        "schema_version": "s2and_arrow_artifact_generation_v1",
                        "generation_id": generation_id,
                        "files": files,
                    }
                }
            ),
            encoding="utf-8",
        )
        return generation_id

    dataset = DummyDataset("validated_arrow_generation", mode="train")
    dataset.arrow_paths = {"signatures": str(signatures_path)}
    dataset.arrow_artifact_generation = publish_generation(b"generation-1")
    dataset.name_tuples = frozenset({("bill", "william")})

    first = feature_port._get_rust_featurizer(dataset)
    dataset.signatures["not-consumed-from-python"] = object()
    second = feature_port._get_rust_featurizer(dataset)
    dataset.arrow_artifact_generation = publish_generation(b"generation-2")
    third = feature_port._get_rust_featurizer(dataset)

    assert second is first
    assert third is not first
    assert DummyRustFeaturizer.created == [dataset.name, dataset.name]


def test_update_rust_cluster_seeds_reuses_cached_featurizer_without_default_version_bump():
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset("direct_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    first = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "c1"

    update_rust_cluster_seeds(dataset)

    assert int(dataset._cluster_seeds_version) == 1
    assert DummyRustFeaturizer.created == ["direct_seed_update_dataset"]
    assert feature_port._get_rust_featurizer(dataset) is first
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


def test_update_rust_cluster_seeds_allows_explicit_version_bump():
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset("explicit_seed_update_bump_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    first = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "c1"

    update_rust_cluster_seeds(dataset, bump_version=True)

    assert int(dataset._cluster_seeds_version) == 2
    assert DummyRustFeaturizer.created == ["explicit_seed_update_bump_dataset"]
    assert feature_port._get_rust_featurizer(dataset) is first
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=2)]


def test_update_rust_cluster_seeds_blocks_cache_prune_until_promotion():
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset("seed_update_promotion_race_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    first = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "c1"
    dataset._cluster_seeds_version = 2
    update_started = threading.Event()
    release_update = threading.Event()
    update_errors: list[Exception] = []
    getter_errors: list[Exception] = []
    getter_results: list[DummyRustFeaturizer] = []

    def blocking_update(_require_map, _disallow_set):
        update_started.set()
        assert release_update.wait(timeout=2)

    first.update_cluster_seeds = blocking_update

    def update_worker():
        try:
            update_rust_cluster_seeds(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            update_errors.append(exc)

    def getter_worker():
        try:
            getter_results.append(feature_port._get_rust_featurizer(dataset))
        except Exception as exc:  # pragma: no cover - assertion guard
            getter_errors.append(exc)

    update_thread = threading.Thread(target=update_worker)
    update_thread.start()
    assert update_started.wait(timeout=2)

    getter_thread = threading.Thread(target=getter_worker)
    getter_thread.start()
    time.sleep(0.05)

    assert getter_results == []
    assert DummyRustFeaturizer.created == ["seed_update_promotion_race_dataset"]

    release_update.set()
    update_thread.join(timeout=5)
    getter_thread.join(timeout=5)

    assert not update_thread.is_alive()
    assert not getter_thread.is_alive()
    assert update_errors == []
    assert getter_errors == []
    assert getter_results == [first]
    assert DummyRustFeaturizer.created == ["seed_update_promotion_race_dataset"]
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=2)]


def test_update_rust_cluster_seeds_leaves_version_unchanged_on_ffi_failure():
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset("failed_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    featurizer = feature_port._get_rust_featurizer(dataset)

    def fail_update(_require_map, _disallow_set):
        raise RuntimeError("ffi failed")

    featurizer.update_cluster_seeds = fail_update

    with pytest.raises(RuntimeError, match="ffi failed"):
        update_rust_cluster_seeds(dataset)

    assert int(dataset._cluster_seeds_version) == 1
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


def test_update_rust_cluster_seeds_rolls_back_featurizer_on_promotion_failure(monkeypatch):
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset("promotion_failure_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    featurizer = feature_port._get_rust_featurizer(dataset)
    featurizer.update_cluster_seeds({"old": "c0"}, {("old", "other")})
    dataset.cluster_seeds_require["s1"] = "c1"

    def fail_promote(_dataset, _featurizer, *, target_seed_version):
        del _dataset, _featurizer, target_seed_version
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(feature_port, "_promote_cached_rust_featurizer_cluster_seed_version", fail_promote)

    with pytest.raises(RuntimeError, match="promotion failed"):
        update_rust_cluster_seeds(dataset, bump_version=True)

    assert int(dataset._cluster_seeds_version) == 1
    assert featurizer.cluster_seeds_require() == [("old", "c0")]
    assert featurizer.cluster_seeds_disallow() == [("old", "other")]


def test_rust_featurizer_cache_rejects_invalid_cluster_seed_version():
    dataset = DummyDataset("bad_seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = "bad"

    with pytest.raises(ValueError, match="invalid literal"):
        feature_port._get_rust_featurizer(dataset)

    assert first.dataset_name == "bad_seed_version_cache_dataset"
    assert DummyRustFeaturizer.created == ["bad_seed_version_cache_dataset"]
    assert _cache_size() == 1


@pytest.mark.parametrize("load_name_counts", [False, True])
def test_build_rust_featurizer_from_arrow_paths_honors_name_count_loading_policy(
    monkeypatch,
    tmp_path,
    load_name_counts: bool,
):
    calls: list[dict[str, Any]] = []

    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, paths, _signature_ids, _name_tuples, *_args):
            calls.append(
                {
                    "paths": dict(paths),
                    "signature_ids": _signature_ids,
                    "name_tuples": _name_tuples,
                    "name_counts_index": _args[4] if len(_args) == 5 else None,
                }
            )
            return cls("arrow")

    class ArrowRustModule:
        __version__ = runtime.REQUIRED_RUST_EXTENSION_VERSION
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)
    canonical_pairs = frozenset({("alice", "ally")})
    monkeypatch.setattr(
        feature_port,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(pairs=canonical_pairs),
    )
    for filename in ("signatures.arrow", "papers.arrow", "paper_authors.arrow"):
        (tmp_path / filename).touch()
    paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
        "signatures_batch_index": str(tmp_path / "signatures.index"),
        "papers_batch_index": str(tmp_path / "papers.index"),
        "paper_authors_batch_index": str(tmp_path / "paper_authors.index"),
    }
    for key in ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index"):
        Path(paths[key]).touch()

    provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path / "name_counts_index", tiny_name_counts_tuple(), provenance)
    complete_paths = {**paths, "name_counts_index": index_path}
    write_test_arrow_artifact_manifest(tmp_path, complete_paths)
    monkeypatch.setattr("s2and.arrow_inputs._validate_batch_indexes", lambda _paths: None)
    validated_paths = validate_arrow_training_artifacts(
        complete_paths,
        require_specter=False,
        require_name_counts_index=True,
        expected_normalization_version=NORMALIZATION_VERSION,
    )
    shared_name_counts_index = (
        SimpleNamespace(
            normalization_version=NORMALIZATION_VERSION,
            name_counts_provenance_binding=(
                provenance["generation_id"],
                provenance["pickle_sha256"],
                provenance["source_snapshot_id"],
                provenance["selected_rows_sha256"],
            ),
        )
        if load_name_counts
        else None
    )
    result = feature_port.build_rust_featurizer_from_arrow_paths(
        validated_paths,
        expected_normalization_version=NORMALIZATION_VERSION,
        signature_ids=[1, "2"],
        load_name_counts=load_name_counts,
        name_counts_index=shared_name_counts_index,
    )

    assert result.dataset_name == "arrow"
    expected_paths = (
        complete_paths
        if load_name_counts
        else {key: value for key, value in complete_paths.items() if key != "name_counts_index"}
    )
    assert calls == [
        {
            "paths": expected_paths,
            "signature_ids": ["1", "2"],
            "name_tuples": canonical_pairs,
            "name_counts_index": shared_name_counts_index,
        }
    ]
    if load_name_counts:
        mismatched_index = SimpleNamespace(
            normalization_version=NORMALIZATION_VERSION,
            name_counts_provenance_binding=(
                "wrong-generation",
                provenance["pickle_sha256"],
                provenance["source_snapshot_id"],
                provenance["selected_rows_sha256"],
            ),
        )
        with pytest.raises(ValueError, match="name-count binding mismatch"):
            feature_port.build_rust_featurizer_from_arrow_paths(
                validated_paths,
                expected_normalization_version=NORMALIZATION_VERSION,
                load_name_counts=True,
                name_counts_index=mismatched_index,
            )


def test_concurrent_builds_for_distinct_datasets_do_not_serialize(monkeypatch):
    d1 = DummyDataset("parallel_d1", mode="train")
    d2 = DummyDataset("parallel_d2", mode="train")
    ready = threading.Event()
    build_windows: dict[str, tuple[float, float]] = {}
    window_lock = threading.Lock()

    def _build_stub(dataset):
        ready.wait(timeout=2)
        build_start = time.perf_counter()
        time.sleep(0.25)
        build_end = time.perf_counter()
        with window_lock:
            build_windows[dataset.name] = (build_start, build_end)
        return (
            DummyRustFeaturizer(dataset.name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.25,
        )

    monkeypatch.setattr(
        feature_port,
        "_build_rust_featurizer_strict",
        _build_stub,
    )

    errors: list[Exception] = []

    def _worker(dataset):
        try:
            feature_port._get_rust_featurizer(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=(d1,))
    t2 = threading.Thread(target=_worker, args=(d2,))
    t1.start()
    t2.start()
    ready.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == []
    assert len(build_windows) == 2
    latest_start = max(window[0] for window in build_windows.values())
    earliest_end = min(window[1] for window in build_windows.values())
    assert latest_start < earliest_end


def test_get_rust_featurizer_raises_after_repeated_empty_wait(monkeypatch):
    dataset = DummyDataset("empty_wait_retry_budget", mode="train")
    attempts = {"count": 0}
    runtime_context = type("RuntimeContext", (), {"operation": "test_empty_wait", "run_id": "run-empty-wait"})()
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 2)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)

    def _always_empty(_dataset, *, build_context):
        del build_context
        attempts["count"] += 1
        return None, None

    monkeypatch.setattr(feature_port, "_get_or_wait_for_cached", _always_empty)

    with pytest.raises(RuntimeError, match="empty wait state") as exc_info:
        feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context)
    message = str(exc_info.value)
    assert "dataset=empty_wait_retry_budget" in message
    assert "run=run-empty-wait" in message
    assert "attempts=3" in message
    assert attempts["count"] == 3


def test_get_rust_featurizer_retries_empty_wait_then_builds(monkeypatch):
    dataset = DummyDataset("empty_wait_then_build", mode="train")
    attempts = {"count": 0}
    build_calls = {"count": 0}
    inflight = feature_port._InFlightFeaturizerBuild()
    expected_featurizer = DummyRustFeaturizer("built_after_empty_wait")
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 2)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)

    def _empty_then_build(_dataset, *, build_context):
        del build_context
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None, None
        return None, inflight

    def _build_stub(_dataset, *, inflight_build, build_context):
        del build_context
        build_calls["count"] += 1
        assert inflight_build is inflight
        return expected_featurizer

    monkeypatch.setattr(feature_port, "_get_or_wait_for_cached", _empty_then_build)
    monkeypatch.setattr(feature_port, "_build_and_cache_rust_featurizer", _build_stub)

    featurizer = feature_port._get_rust_featurizer(dataset)
    assert featurizer is expected_featurizer
    assert attempts["count"] == 2
    assert build_calls["count"] == 1


def test_get_rust_featurizer_raises_after_repeated_stale_build(monkeypatch):
    dataset = DummyDataset("stale_build_retry_budget", mode="train")
    runtime_context = type("RuntimeContext", (), {"operation": "test_stale_build", "run_id": "run-stale-build"})()
    inflight = feature_port._InFlightFeaturizerBuild()
    build_calls = {"count": 0}
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 2)

    def _always_build(_dataset, *, build_context):
        del build_context
        return None, inflight

    def _always_stale(_dataset, *, inflight_build, build_context):
        del build_context
        assert inflight_build is inflight
        build_calls["count"] += 1
        return None

    monkeypatch.setattr(feature_port, "_get_or_wait_for_cached", _always_build)
    monkeypatch.setattr(feature_port, "_build_and_cache_rust_featurizer", _always_stale)

    with pytest.raises(RuntimeError, match="stale build state") as exc_info:
        feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context)

    message = str(exc_info.value)
    assert "dataset=stale_build_retry_budget" in message
    assert "run=run-stale-build" in message
    assert "attempts=3" in message
    assert build_calls["count"] == 3


def test_explicit_evict_and_clear_api():
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    feature_port._get_rust_featurizer(d1)
    feature_port._get_rust_featurizer(d2)
    assert _cache_size() == 2

    assert feature_port.evict_rust_featurizer(d1) is True
    assert feature_port.evict_rust_featurizer(d1) is False
    assert _cache_size() == 1

    cleared = feature_port.clear_rust_featurizer_cache()
    assert cleared == 1
    assert _cache_size() == 0


def test_evict_during_inflight_build_discards_stale_result(monkeypatch):
    dataset = DummyDataset("evict_inflight", mode="train")
    first_build_started = threading.Event()
    release_first_build = threading.Event()
    build_calls = {"count": 0}

    def _build_stub(dataset_arg):
        build_calls["count"] += 1
        if build_calls["count"] == 1:
            first_build_started.set()
            release_first_build.wait(timeout=2)
            return (
                DummyRustFeaturizer(f"{dataset_arg.name}_stale"),
                {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
                0.0,
            )
        return (
            DummyRustFeaturizer(f"{dataset_arg.name}_fresh"),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)
    results: list[DummyRustFeaturizer] = []
    errors: list[Exception] = []

    def _worker():
        try:
            results.append(feature_port._get_rust_featurizer(dataset))
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    assert first_build_started.wait(timeout=2)
    assert feature_port.evict_rust_featurizer(dataset) is False
    release_first_build.set()
    thread.join(timeout=5)

    assert errors == []
    assert [result.dataset_name for result in results] == ["evict_inflight_fresh"]
    assert build_calls["count"] == 2
    assert feature_port._get_rust_featurizer(dataset).dataset_name == "evict_inflight_fresh"


def test_clear_during_inflight_build_discards_stale_result(monkeypatch):
    dataset = DummyDataset("clear_inflight", mode="train")
    first_build_started = threading.Event()
    release_first_build = threading.Event()
    build_calls = {"count": 0}

    def _build_stub(dataset_arg):
        build_calls["count"] += 1
        if build_calls["count"] == 1:
            first_build_started.set()
            release_first_build.wait(timeout=2)
            return (
                DummyRustFeaturizer(f"{dataset_arg.name}_stale"),
                {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
                0.0,
            )
        return (
            DummyRustFeaturizer(f"{dataset_arg.name}_fresh"),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)
    results: list[DummyRustFeaturizer] = []
    errors: list[Exception] = []

    def _worker():
        try:
            results.append(feature_port._get_rust_featurizer(dataset))
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    assert first_build_started.wait(timeout=2)
    assert feature_port.clear_rust_featurizer_cache() == 0
    release_first_build.set()
    thread.join(timeout=5)

    assert errors == []
    assert [result.dataset_name for result in results] == ["clear_inflight_fresh"]
    assert build_calls["count"] == 2
    assert feature_port._get_rust_featurizer(dataset).dataset_name == "clear_inflight_fresh"


def test_evict_rust_featurizer_clears_build_counts():
    dataset = DummyDataset("evict_build_counts", mode="train")
    cache_key = feature_port._rust_featurizer_cache_key(dataset)  # noqa: SLF001

    feature_port._get_rust_featurizer(dataset)

    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == 1  # noqa: SLF001
    assert feature_port.evict_rust_featurizer(dataset) is True
    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == 0  # noqa: SLF001
