import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.feature_port as feature_port
import s2and.runtime as runtime
from s2and.arrow_inputs import MissingArrowArtifactError
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import patch_tiny_name_counts_loader

# Derived from the guard so fixtures track the lockstep version bump instead of going stale.
_SUPPORTED_VERSION = runtime.min_supported_rust_extension_version_string()


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


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
    def from_dataset(cls, dataset, _require_value, _disallow_value, _num_threads=None):
        cls.created.append(dataset.name)
        return cls(dataset.name)

    @classmethod
    def from_arrow_paths(cls, *_args, **_kwargs):
        return cls("arrow")

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
    __version__ = _SUPPORTED_VERSION
    RustFeaturizer = DummyRustFeaturizer


class DummyDataset(ANDData):
    def __init__(self, name: str, mode: str = "train"):
        self.name = name
        self.mode = mode
        self.signatures = {}
        self.papers = {}
        self.name_tuples = set()
        self.compute_reference_features = False
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
        self.name_counts_last_first_initial_semantics = "legacy_compat"


class SleepyCounter:
    """Counter that yields mid-increment to expose race windows in tests."""

    def __init__(self, value: int = 0):
        self.value = value

    def __iadd__(self, increment: int):
        current = self.value
        time.sleep(0.0005)
        self.value = current + int(increment)
        return self

    def __int__(self) -> int:
        return self.value


def _cache_size() -> int:
    return sum(len(entries) for entries in feature_port._RUST_FEATURIZER_CACHE.values())


def _cache_keys(dataset: DummyDataset) -> list[feature_port._RustFeaturizerCacheKey]:
    return list(feature_port._RUST_FEATURIZER_CACHE[dataset])


@pytest.fixture(autouse=True)
def _reset_feature_port_state(monkeypatch):
    feature_port.clear_rust_featurizer_cache()
    DummyRustFeaturizer.created = []
    DummyRustFeaturizer.signature_overlay_payloads = []
    monkeypatch.setattr(feature_port, "s2and_rust", DummyRustModule)
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


def test_rust_featurizer_available_reloads_extension_when_missing(monkeypatch):
    sentinel_module = object()
    load_calls = {"count": 0}

    def _load_stub():
        load_calls["count"] += 1
        return sentinel_module

    def _detect_stub(*, extension_module):
        return SimpleNamespace(core_runtime_available=extension_module is sentinel_module)

    monkeypatch.setattr(feature_port, "s2and_rust", None)
    monkeypatch.setattr(feature_port, "load_s2and_rust_extension", _load_stub)
    monkeypatch.setattr(feature_port, "detect_rust_runtime_capabilities", _detect_stub)

    assert feature_port.rust_featurizer_available() is True
    assert feature_port.s2and_rust is sentinel_module
    assert load_calls["count"] == 1


def test_rust_featurizer_cache_tracks_cluster_seed_version():
    dataset = DummyDataset("seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = 1
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == ["seed_version_cache_dataset", "seed_version_cache_dataset"]
    assert _cache_size() == 1
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)]


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
        ("compute_reference_features", lambda dataset: setattr(dataset, "compute_reference_features", True)),
        ("preprocess", lambda dataset: setattr(dataset, "preprocess", False)),
        ("n_jobs", lambda dataset: setattr(dataset, "n_jobs", 2)),
        ("signatures_path", lambda dataset: setattr(dataset, "signatures_path", "new_signatures.json")),
        ("signatures", lambda dataset: dataset.signatures.__setitem__("s1", object())),
        ("papers", lambda dataset: dataset.papers.__setitem__("p1", object())),
        ("specter_embeddings", lambda dataset: setattr(dataset, "specter_embeddings", {"p1": object()})),
        ("name_tuples", lambda dataset: dataset.name_tuples.add(("bill", "william"))),
    ],
)
def test_rust_featurizer_cache_tracks_material_from_dataset_fields(case_name, mutate_dataset):
    dataset = DummyDataset(f"material_cache_{case_name}", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    mutate_dataset(dataset)
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == [dataset.name, dataset.name]
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset)]


def test_rust_featurizer_cache_tracks_material_mutation_beyond_prefix_sample():
    dataset = DummyDataset("material_cache_full_digest", mode="train")
    dataset.signatures = {f"s{index}": object() for index in range(64)}

    first = feature_port._get_rust_featurizer(dataset)
    dataset.signatures["s63"] = object()
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == [dataset.name, dataset.name]
    assert _cache_keys(dataset) == [feature_port._rust_featurizer_cache_key(dataset)]


def test_rust_featurizer_cache_tracks_in_place_numpy_embedding_mutation(monkeypatch):
    class EmbeddingRustFeaturizer(DummyRustFeaturizer):
        snapshots: list[float] = []

        @classmethod
        def from_dataset(cls, dataset, _require_value, _disallow_value, _num_threads=None):
            cls.snapshots.append(float(dataset.specter_embeddings["p"][0]))
            return cls(dataset.name)

    class EmbeddingRustModule:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = EmbeddingRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", EmbeddingRustModule)
    dataset = DummyDataset("embedding_mutation_cache_dataset", mode="train")
    dataset.specter_embeddings = {"p": np.asarray([0.0], dtype=np.float32)}

    first = feature_port._get_rust_featurizer(dataset)
    dataset.specter_embeddings["p"][0] = 1.0
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert EmbeddingRustFeaturizer.snapshots == [0.0, 1.0]


def test_update_rust_cluster_seeds_reuses_cached_featurizer_without_default_version_bump():
    from s2and.rust_calls import update_rust_cluster_seeds

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
    from s2and.rust_calls import update_rust_cluster_seeds

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
    from s2and.rust_calls import update_rust_cluster_seeds

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
    from s2and.rust_calls import update_rust_cluster_seeds

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
    from s2and import rust_calls
    from s2and.rust_calls import update_rust_cluster_seeds

    dataset = DummyDataset("promotion_failure_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    featurizer = feature_port._get_rust_featurizer(dataset)
    featurizer.update_cluster_seeds({"old": "c0"}, {("old", "other")})
    dataset.cluster_seeds_require["s1"] = "c1"

    def fail_promote(_dataset, _featurizer, *, target_seed_version):
        del _dataset, _featurizer, target_seed_version
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(rust_calls, "_promote_rust_featurizer_cluster_seed_version", fail_promote)

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


def test_build_rust_featurizer_from_arrow_paths_rejects_none_path(monkeypatch):
    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, *_args, **_kwargs):
            raise AssertionError("from_arrow_paths should not be called for invalid paths")

    class ArrowRustModule:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)

    with pytest.raises(ValueError, match="papers.*None"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                "signatures": "signatures.arrow",
                "papers": None,
            }
        )


def test_build_rust_featurizer_from_arrow_paths_requires_index_for_name_counts(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []

    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, paths, _signature_ids, _name_tuples, *_args):
            calls.append({"paths": dict(paths)})
            return cls("arrow")

    class ArrowRustModule:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)
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

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        feature_port.build_rust_featurizer_from_arrow_paths(
            paths,
            load_name_counts=True,
        )
    assert exc_info.value.missing_keys == ("name_counts_index",)

    patch_tiny_name_counts_loader(monkeypatch)
    index_path, _metrics = write_name_counts_index(tmp_path / "name_counts_index")
    result = feature_port.build_rust_featurizer_from_arrow_paths(
        {**paths, "name_counts_index": index_path},
        load_name_counts=True,
    )

    assert result.dataset_name == "arrow"
    assert calls == [
        {
            "paths": {**paths, "name_counts_index": index_path},
        }
    ]


def test_build_rust_featurizer_from_arrow_paths_requires_batch_indexes_by_default(monkeypatch, tmp_path):
    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, *_args, **_kwargs):
            return cls("arrow")

    class ArrowRustModule:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)
    paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
    }
    for path in paths.values():
        Path(path).touch()

    with pytest.raises(ValueError, match="signatures_batch_index"):
        feature_port.build_rust_featurizer_from_arrow_paths(paths)

    with pytest.raises(ValueError, match="missing_papers.arrow"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                **paths,
                "papers": str(tmp_path / "missing_papers.arrow"),
                "signatures_batch_index": str(tmp_path / "signatures.index"),
                "papers_batch_index": str(tmp_path / "papers.index"),
                "paper_authors_batch_index": str(tmp_path / "paper_authors.index"),
            },
        )

    with pytest.raises(ValueError, match="name_pairs"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                **paths,
                "name_pairs": str(tmp_path / "missing_name_pairs.arrow"),
                "signatures_batch_index": str(tmp_path / "signatures.index"),
                "papers_batch_index": str(tmp_path / "papers.index"),
                "paper_authors_batch_index": str(tmp_path / "paper_authors.index"),
            },
        )

    index_paths = {
        "signatures_batch_index": str(tmp_path / "signatures.index"),
        "papers_batch_index": str(tmp_path / "papers.index"),
        "paper_authors_batch_index": str(tmp_path / "paper_authors.index"),
    }
    for path in index_paths.values():
        Path(path).touch()

    result = feature_port.build_rust_featurizer_from_arrow_paths({**paths, **index_paths})
    assert result.dataset_name == "arrow"


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


def test_concurrent_builds_for_same_dataset_share_single_inflight_build(monkeypatch):
    dataset = DummyDataset("parallel_same_dataset", mode="train")
    ready = threading.Event()
    build_calls = {"count": 0}

    def _build_stub(dataset_arg):
        build_calls["count"] += 1
        ready.wait(timeout=2)
        time.sleep(0.25)
        return (
            DummyRustFeaturizer(dataset_arg.name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.25,
        )

    monkeypatch.setattr(
        feature_port,
        "_build_rust_featurizer_strict",
        _build_stub,
    )

    errors: list[Exception] = []

    def _worker():
        try:
            feature_port._get_rust_featurizer(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    ready.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == []
    assert build_calls["count"] == 1


def test_increment_rust_featurizer_build_count_is_thread_safe():
    dataset = DummyDataset("build_count_threadsafe", mode="train")
    cache_key = feature_port._rust_featurizer_cache_key(dataset)  # noqa: SLF001
    with feature_port._RUST_FEATURIZER_CACHE_LOCK:
        feature_port._RUST_FEATURIZER_CACHE[dataset] = {
            cache_key: feature_port._CacheEntry(
                featurizer=DummyRustFeaturizer(dataset.name),
                build_count=cast(Any, SleepyCounter(0)),
            )
        }

    worker_count = 8
    increments_per_worker = 25
    errors: list[Exception] = []
    counts: list[int] = []
    counts_lock = threading.Lock()

    def _worker():
        try:
            for _ in range(increments_per_worker):
                next_count = feature_port._increment_rust_featurizer_build_count(dataset, cache_key)
                with counts_lock:
                    counts.append(next_count)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    expected_count = worker_count * increments_per_worker
    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == expected_count
    assert sorted(counts) == list(range(1, expected_count + 1))


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


@pytest.mark.parametrize(
    ("mode", "with_json_paths"),
    [
        ("train", True),
        ("inference", False),
        ("inference", True),
    ],
)
def test_rust_featurizer_build_uses_dataset_constructor(mode: str, with_json_paths: bool):
    dataset = DummyDataset(f"{mode}_paths{int(with_json_paths)}", mode=mode)
    if with_json_paths:
        dataset.signatures_path = "signatures.json"
        dataset.papers_path = "papers.json"

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == [dataset.name]


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


def test_load_s2and_rust_extension_falls_back_from_namespace_package(monkeypatch):
    class NamespaceOnly:
        pass

    class NativeExtension:
        RustFeaturizer = object()

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return NamespaceOnly()
        if name == "s2and_rust._s2and_rust":
            return NativeExtension
        raise _missing_module(name)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)

    loaded = runtime.load_s2and_rust_extension()
    assert loaded is NativeExtension


def test_load_s2and_rust_extension_returns_first_valid_module(monkeypatch):
    class ValidRustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

    class TopLevelModule:
        RustFeaturizer = ValidRustFeaturizer

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return TopLevelModule
        raise _missing_module(name)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)

    loaded = runtime.load_s2and_rust_extension()
    assert loaded is TopLevelModule
