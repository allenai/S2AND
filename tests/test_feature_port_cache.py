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
import s2and.model as model_module
import s2and.runtime as runtime
from s2and.arrow_inputs import validate_arrow_training_artifacts
from s2and.consts import NORMALIZATION_VERSION
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.runtime import RuntimeContext
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
    return sum(state.entry is not None for state in feature_port._RUST_FEATURIZER_STATES.values())


def _cached_entry(dataset: DummyDataset) -> feature_port._CacheEntry | None:
    state = feature_port._RUST_FEATURIZER_STATES.get(dataset)
    return None if state is None else state.entry


def _cache_keys(dataset: DummyDataset) -> list[feature_port._RustFeaturizerCacheKey]:
    entry = _cached_entry(dataset)
    assert entry is not None
    return [entry.cache_key]


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
    assert _cached_entry(d1) is not None
    assert _cached_entry(d2) is not None

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


def test_rust_featurizer_cache_key_rechecks_build_inputs(monkeypatch):
    dataset = DummyDataset("memoized_non_seed_cache_dataset", mode="train")
    calls = {"count": 0}

    def fake_source_paths(_dataset):
        calls["count"] += 1
        return ()

    monkeypatch.setattr(feature_port, "_rust_featurizer_source_paths", fake_source_paths)

    first_key = feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=0)
    second_key = feature_port._rust_featurizer_cache_key(dataset, cluster_seeds_version=1)

    assert calls["count"] == 2
    assert first_key.build == second_key.build
    assert first_key.cluster_seeds_version == 0
    assert second_key.cluster_seeds_version == 1


def test_rust_featurizer_rejects_inputs_changing_during_build(monkeypatch):
    dataset = DummyDataset("seed_version_build_race_dataset", mode="train")
    dataset._cluster_seeds_version = 0

    def _build_stub(dataset_arg):
        dataset_arg._cluster_seeds_version = 1
        return (
            DummyRustFeaturizer("stale"),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)

    with pytest.raises(RuntimeError, match="inputs changed while it was being built"):
        feature_port._get_rust_featurizer(dataset)

    assert _cached_entry(dataset) is None


@pytest.mark.parametrize(
    ("case_name", "mutate_dataset"),
    [
        ("preprocess", lambda dataset: setattr(dataset, "preprocess", False)),
        ("use_orcid_id", lambda dataset: setattr(dataset, "use_orcid_id", False)),
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


@pytest.mark.parametrize("cache_action", ["evict", "clear"])
def test_cache_eviction_waits_for_cluster_seed_update(cache_action):
    from s2and.feature_port import update_rust_cluster_seeds

    dataset = DummyDataset(f"seed_update_{cache_action}_race_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    featurizer = feature_port._get_rust_featurizer(dataset)
    featurizer.update_cluster_seeds({"old": "c-old"}, {("old", "other")})
    dataset.cluster_seeds_require = {"new": "c-new"}
    dataset.cluster_seeds_disallow = {("new", "other")}
    update_started = threading.Event()
    release_update = threading.Event()
    update_errors: list[Exception] = []
    eviction_results: list[bool | int] = []
    original_update = featurizer.update_cluster_seeds

    def blocking_update(require, disallow):
        update_started.set()
        assert release_update.wait(timeout=2)
        original_update(require, disallow)

    featurizer.update_cluster_seeds = blocking_update

    def update_worker():
        try:
            update_rust_cluster_seeds(dataset, bump_version=True)
        except Exception as exc:
            update_errors.append(exc)

    update_thread = threading.Thread(target=update_worker)
    update_thread.start()
    assert update_started.wait(timeout=2)

    def evict_worker():
        if cache_action == "evict":
            eviction_results.append(feature_port.evict_rust_featurizer(dataset))
        else:
            eviction_results.append(feature_port.clear_rust_featurizer_cache())

    eviction_thread = threading.Thread(target=evict_worker)
    eviction_thread.start()
    time.sleep(0.05)
    assert eviction_thread.is_alive()
    assert _cached_entry(dataset) is not None

    release_update.set()
    update_thread.join(timeout=5)
    eviction_thread.join(timeout=5)

    assert not update_thread.is_alive()
    assert not eviction_thread.is_alive()
    assert update_errors == []
    assert eviction_results == [True if cache_action == "evict" else 1]
    assert int(dataset._cluster_seeds_version) == 2
    assert featurizer.cluster_seeds_require() == [("new", "c-new")]
    assert featurizer.cluster_seeds_disallow() == [("new", "other")]
    assert _cached_entry(dataset) is None


@pytest.mark.parametrize("cache_action", ["evict", "clear"])
def test_successful_seed_sync_survives_later_cache_eviction(monkeypatch, cache_action):
    dataset = DummyDataset(f"synced_seed_{cache_action}_rebuild_dataset", mode="train")
    dataset.runtime_context = RuntimeContext(operation="constraints", backend="rust", run_id="seed-rebuild")
    dataset._cluster_seeds_version = 1
    dataset.cluster_seeds_require = {"current": "c-current"}
    dataset.cluster_seeds_disallow = {("current", "other")}

    def build_with_immutable_arrow_seeds(dataset_arg):
        DummyRustFeaturizer.created.append(dataset_arg.name)
        featurizer = DummyRustFeaturizer(dataset_arg.name)
        featurizer.update_cluster_seeds({"arrow": "c-arrow"}, {("arrow", "other")})
        return (
            featurizer,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
        )

    monkeypatch.setattr(model_module, "_use_rust_constraints", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(feature_port, "build_rust_featurizer", build_with_immutable_arrow_seeds)

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=dataset.runtime_context)
    first = feature_port._get_rust_featurizer(dataset)
    assert first.cluster_seeds_require() == [("current", "c-current")]
    assert first.cluster_seeds_disallow() == [("current", "other")]

    if cache_action == "evict":
        assert feature_port.evict_rust_featurizer(dataset)
    else:
        assert feature_port.clear_rust_featurizer_cache() == 1
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=dataset.runtime_context)
    assert dataset._rust_cluster_seeds_sync_skipped_unchanged == 1

    rebuilt = feature_port._get_rust_featurizer(dataset)
    assert rebuilt is not first
    assert rebuilt.cluster_seeds_require() == [("current", "c-current")]
    assert rebuilt.cluster_seeds_disallow() == [("current", "other")]


def test_arrow_authoritative_seeds_survive_initial_sync_and_cache_rebuild(monkeypatch):
    dataset = DummyDataset("arrow_seed_authority_dataset", mode="train")
    dataset.runtime_context = RuntimeContext(operation="constraints", backend="rust", run_id="arrow-seed-authority")
    dataset._cluster_seeds_version = 1
    dataset._rust_cluster_seeds_synced_version = 0
    dataset._cluster_seeds_source = "arrow"

    def build_with_immutable_arrow_seeds(dataset_arg):
        DummyRustFeaturizer.created.append(dataset_arg.name)
        featurizer = DummyRustFeaturizer(dataset_arg.name)
        featurizer.update_cluster_seeds({"arrow": "c-arrow"}, {("arrow", "other")})
        return (
            featurizer,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
        )

    monkeypatch.setattr(model_module, "_use_rust_constraints", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(feature_port, "build_rust_featurizer", build_with_immutable_arrow_seeds)

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=dataset.runtime_context)
    first = feature_port._get_rust_featurizer(dataset)
    assert first.cluster_seeds_require() == [("arrow", "c-arrow")]
    assert first.cluster_seeds_disallow() == [("arrow", "other")]

    assert feature_port.evict_rust_featurizer(dataset)
    rebuilt = feature_port._get_rust_featurizer(dataset)
    assert rebuilt.cluster_seeds_require() == [("arrow", "c-arrow")]
    assert rebuilt.cluster_seeds_disallow() == [("arrow", "other")]

    dataset.cluster_seeds_require = {"python": "c-python"}
    dataset.cluster_seeds_disallow = {("python", "other")}
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=dataset.runtime_context)

    assert dataset._cluster_seeds_source == "python"
    assert rebuilt.cluster_seeds_require() == [("python", "c-python")]
    assert rebuilt.cluster_seeds_disallow() == [("python", "other")]


@pytest.mark.parametrize("replace_with_empty", [False, True])
def test_python_seed_assignment_before_first_arrow_sync_takes_authority(monkeypatch, replace_with_empty):
    dataset = DummyDataset(f"pre_sync_python_seed_{replace_with_empty}", mode="train")
    dataset.runtime_context = RuntimeContext(operation="constraints", backend="rust", run_id="pre-sync-python-seed")
    dataset._cluster_seeds_version = 1
    dataset._rust_cluster_seeds_synced_version = 0
    dataset._cluster_seeds_source = "arrow"
    dataset._cluster_seeds_initial_require_id = id(dataset.cluster_seeds_require)
    dataset._cluster_seeds_initial_disallow_id = id(dataset.cluster_seeds_disallow)

    def build_with_immutable_arrow_seeds(dataset_arg):
        featurizer = DummyRustFeaturizer(dataset_arg.name)
        featurizer.update_cluster_seeds({"arrow": "c-arrow"}, {("arrow", "other")})
        return (
            featurizer,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
        )

    monkeypatch.setattr(model_module, "_use_rust_constraints", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(feature_port, "build_rust_featurizer", build_with_immutable_arrow_seeds)

    if replace_with_empty:
        dataset.cluster_seeds_require = {}
        dataset.cluster_seeds_disallow = set()
        expected_require = []
        expected_disallow = []
    else:
        dataset.cluster_seeds_require["python"] = "c-python"
        dataset.cluster_seeds_disallow.add(("python", "other"))
        expected_require = [("python", "c-python")]
        expected_disallow = [("python", "other")]

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=dataset.runtime_context)
    featurizer = feature_port._get_rust_featurizer(dataset)

    assert dataset._cluster_seeds_source == "python"
    assert featurizer.cluster_seeds_require() == expected_require
    assert featurizer.cluster_seeds_disallow() == expected_disallow


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
@pytest.mark.parametrize("use_orcid_id", [False, True])
def test_build_rust_featurizer_from_arrow_paths_honors_name_count_loading_policy(
    monkeypatch,
    tmp_path,
    load_name_counts: bool,
    use_orcid_id: bool,
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
                    "name_counts_index": _args[4] if len(_args) >= 5 else None,
                    "use_orcid_id": _args[5] if len(_args) >= 6 else True,
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
    monkeypatch.setattr(
        "s2and.arrow_inputs._validate_batch_indexes",
        lambda _paths, _generation_files, **_kwargs: None,
    )
    validated_paths = validate_arrow_training_artifacts(
        complete_paths,
        require_specter=False,
        require_name_counts_index=True,
        expected_normalization_version=NORMALIZATION_VERSION,
    )
    validated_manifest = validated_paths.name_counts_manifest
    assert validated_manifest is not None
    shared_name_counts_index = (
        SimpleNamespace(
            manifest_sha256=validated_manifest.manifest_sha256,
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
        use_orcid_id=use_orcid_id,
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
            "use_orcid_id": use_orcid_id,
        }
    ]
    if load_name_counts:
        mismatched_index = SimpleNamespace(
            manifest_sha256=validated_manifest.manifest_sha256,
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


def test_concurrent_builds_for_same_dataset_build_once(monkeypatch):
    dataset = DummyDataset("same_dataset", mode="train")
    build_started = threading.Event()
    release_build = threading.Event()
    build_calls = 0

    def _build_stub(dataset_arg):
        nonlocal build_calls
        build_calls += 1
        build_started.set()
        assert release_build.wait(timeout=2)
        return (
            DummyRustFeaturizer(dataset_arg.name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)
    results: list[DummyRustFeaturizer] = []

    threads = [
        threading.Thread(target=lambda: results.append(feature_port._get_rust_featurizer(dataset))) for _ in range(2)
    ]
    threads[0].start()
    assert build_started.wait(timeout=2)
    threads[1].start()
    time.sleep(0.05)
    assert build_calls == 1

    release_build.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert build_calls == 1
    assert len(results) == 2
    assert results[0] is results[1]


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


@pytest.mark.parametrize("cache_action", ["evict", "clear"])
def test_cache_eviction_waits_for_build_then_removes_entry(monkeypatch, cache_action):
    dataset = DummyDataset(f"{cache_action}_inflight", mode="train")
    first_build_started = threading.Event()
    release_first_build = threading.Event()

    def _build_stub(dataset_arg):
        first_build_started.set()
        assert release_first_build.wait(timeout=2)
        return (
            DummyRustFeaturizer(dataset_arg.name),
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)
    results: list[DummyRustFeaturizer] = []
    eviction_results: list[bool | int] = []

    build_thread = threading.Thread(target=lambda: results.append(feature_port._get_rust_featurizer(dataset)))
    build_thread.start()
    assert first_build_started.wait(timeout=2)

    def evict_worker():
        if cache_action == "evict":
            eviction_results.append(feature_port.evict_rust_featurizer(dataset))
        else:
            eviction_results.append(feature_port.clear_rust_featurizer_cache())

    eviction_thread = threading.Thread(target=evict_worker)
    eviction_thread.start()
    time.sleep(0.05)
    assert eviction_thread.is_alive()
    release_first_build.set()
    build_thread.join(timeout=5)
    eviction_thread.join(timeout=5)

    assert not build_thread.is_alive()
    assert not eviction_thread.is_alive()
    assert [result.dataset_name for result in results] == [dataset.name]
    assert eviction_results == [True if cache_action == "evict" else 1]
    assert _cached_entry(dataset) is None
