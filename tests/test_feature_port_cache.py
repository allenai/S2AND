import threading
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.feature_port as feature_port
import s2and.runtime as runtime
from s2and.data import ANDData
from tests.helpers import tiny_name_counts_provenance


class DummyRustFeaturizer:
    created: list[str] = []

    def __init__(
        self,
        dataset_name: str,
        require: dict[Any, Any] | None = None,
        disallow: set[tuple[Any, Any]] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.require = dict(require or {})
        self.disallow = set(disallow or set())
        self.update_attempts = 0
        self.update_payloads: list[tuple[dict[Any, Any], set[tuple[Any, Any]]]] = []
        self.fail_updates = False

    def update_cluster_seeds(
        self,
        require: dict[Any, Any],
        disallow: set[tuple[Any, Any]],
    ) -> None:
        self.update_attempts += 1
        if self.fail_updates:
            raise RuntimeError("seed update failed")
        self.require = dict(require)
        self.disallow = set(disallow)
        self.update_payloads.append((self.require, self.disallow))


class DummyRustModule:
    __version__ = runtime.REQUIRED_RUST_EXTENSION_VERSION
    RustFeaturizer = DummyRustFeaturizer


class DummyDataset(ANDData):
    def __init__(self, name: str, *, seed_source: str = "arrow") -> None:
        self.name = name
        self.mode = "train"
        self.runtime_context = SimpleNamespace(operation="test", run_id=name)
        self.signatures: dict[str, Any] = {}
        self.papers: dict[str, Any] = {}
        self.name_tuples: set[tuple[str, str]] = set()
        self.preprocess = True
        self.use_orcid_id = True
        self.n_jobs = 1
        self.cluster_seeds_require: dict[Any, Any] = {}
        self.cluster_seeds_disallow: set[tuple[Any, Any]] = set()
        self._cluster_seeds_source = seed_source
        self._cluster_seeds_initial_require_id = id(self.cluster_seeds_require)
        self._cluster_seeds_initial_disallow_id = id(self.cluster_seeds_disallow)
        self.arrow_seed_require: dict[Any, Any] = {}
        self.arrow_seed_disallow: set[tuple[Any, Any]] = set()
        self.arrow_paths = {"signatures": f"{name}.arrow"}
        self.arrow_artifact_generation = f"test-generation-{name}"
        self.name_counts_provenance = tiny_name_counts_provenance()


def _dummy_build_rust_featurizer(
    dataset: DummyDataset,
) -> tuple[DummyRustFeaturizer, dict[str, float]]:
    DummyRustFeaturizer.created.append(dataset.name)
    return (
        DummyRustFeaturizer(
            dataset.name,
            dataset.arrow_seed_require,
            dataset.arrow_seed_disallow,
        ),
        {
            "pre_build_seconds": 0.0,
            "ffi_seconds": 0.0,
            "post_build_seconds": 0.0,
        },
    )


@pytest.fixture(autouse=True)
def _use_dummy_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    DummyRustFeaturizer.created = []
    monkeypatch.setattr(feature_port, "s2and_rust", DummyRustModule)
    monkeypatch.setattr(
        feature_port,
        "build_rust_featurizer",
        _dummy_build_rust_featurizer,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__init__({"replacement": 3}),
        lambda value: value.__setitem__("a", 3),
        lambda value: value.__delitem__("a"),
        lambda value: value.__ior__({"replacement": 3}),
        lambda value: value.clear(),
        lambda value: value.pop("a"),
        lambda value: value.popitem(),
        lambda value: value.setdefault("replacement", 3),
        lambda value: value.update({"replacement": 3}),
    ],
    ids=[
        "init",
        "setitem",
        "delitem",
        "ior",
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
    ],
)
def test_mutation_tracked_dict_covers_builtin_mutators(mutate: Any) -> None:
    value = feature_port._MutationTrackedDict({"a": 1, "b": 2})
    initial_version = value.mutation_version

    mutate(value)

    assert value.mutation_version > initial_version
    assert not hasattr(value, "__dict__")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__init__({("replacement", "pair")}),
        lambda value: value.__iand__({("a", "b")}),
        lambda value: value.__ior__({("replacement", "pair")}),
        lambda value: value.__isub__({("a", "b")}),
        lambda value: value.__ixor__({("replacement", "pair")}),
        lambda value: value.add(("replacement", "pair")),
        lambda value: value.clear(),
        lambda value: value.difference_update({("a", "b")}),
        lambda value: value.discard(("a", "b")),
        lambda value: value.intersection_update({("a", "b")}),
        lambda value: value.pop(),
        lambda value: value.remove(("a", "b")),
        lambda value: value.symmetric_difference_update({("replacement", "pair")}),
        lambda value: value.update({("replacement", "pair")}),
    ],
    ids=[
        "init",
        "iand",
        "ior",
        "isub",
        "ixor",
        "add",
        "clear",
        "difference-update",
        "discard",
        "intersection-update",
        "pop",
        "remove",
        "symmetric-difference-update",
        "update",
    ],
)
def test_mutation_tracked_set_covers_builtin_mutators(mutate: Any) -> None:
    value = feature_port._MutationTrackedSet({("a", "b"), ("c", "d")})
    initial_version = value.mutation_version

    mutate(value)

    assert value.mutation_version > initial_version
    assert not hasattr(value, "__dict__")


def test_tracked_containers_ignore_common_no_op_mutators() -> None:
    require = feature_port._MutationTrackedDict({"s1": "c1"})
    disallow = feature_port._MutationTrackedSet({("s1", "s2")})
    require_version = require.mutation_version
    disallow_version = disallow.mutation_version

    require["s1"] = "c1"
    require |= {}
    require.pop("missing", None)
    require.setdefault("s1", "replacement")
    require.update({})
    disallow &= disallow
    disallow |= set()
    disallow -= set()
    disallow ^= set()
    disallow.add(("s1", "s2"))
    disallow.difference_update(set())
    disallow.discard(("missing", "pair"))
    disallow.intersection_update()
    disallow.symmetric_difference_update(set())
    disallow.update()

    assert require.mutation_version == require_version
    assert disallow.mutation_version == disallow_version
    empty_require = feature_port._MutationTrackedDict()
    empty_disallow = feature_port._MutationTrackedSet()
    empty_require.clear()
    empty_disallow.clear()
    assert empty_require.mutation_version == 0
    assert empty_disallow.mutation_version == 0


def test_cache_is_owned_by_each_dataset_and_evicts() -> None:
    first_dataset = DummyDataset("first")
    second_dataset = DummyDataset("second")

    first = feature_port._get_rust_featurizer(first_dataset)
    second = feature_port._get_rust_featurizer(second_dataset)

    assert feature_port._get_rust_featurizer(first_dataset) is first
    assert feature_port._get_rust_featurizer(second_dataset) is second
    assert DummyRustFeaturizer.created == ["first", "second"]
    assert feature_port._rust_featurizer_build_count(first_dataset) == 1
    assert feature_port._rust_featurizer_build_count(second_dataset) == 1

    assert feature_port.evict_rust_featurizer(first_dataset)
    assert not feature_port.evict_rust_featurizer(first_dataset)
    assert feature_port._rust_featurizer_build_count(first_dataset) == 0
    assert feature_port._get_rust_featurizer(first_dataset) is not first
    assert feature_port._get_rust_featurizer(second_dataset) is second


@pytest.mark.parametrize(
    "attribute",
    [
        "arrow_paths",
        "name_tuples",
        "preprocess",
        "use_orcid_id",
        "n_jobs",
    ],
)
def test_missing_mandatory_build_input_fails_fast(attribute: str) -> None:
    dataset = DummyDataset("missing-build-input")
    delattr(dataset, attribute)

    with pytest.raises(AttributeError, match=attribute):
        feature_port._rust_featurizer_build_inputs(dataset)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda dataset: setattr(dataset, "preprocess", False),
        lambda dataset: setattr(dataset, "use_orcid_id", False),
        lambda dataset: setattr(dataset, "n_jobs", 2),
        lambda dataset: dataset.name_tuples.add(("bill", "william")),
        lambda dataset: setattr(dataset, "arrow_artifact_generation", "replacement"),
        lambda dataset: dataset.arrow_paths.__setitem__("signatures", "replacement.arrow"),
    ],
)
def test_material_build_input_changes_rebuild(
    mutate: Any,
) -> None:
    dataset = DummyDataset("material-input")

    first = feature_port._get_rust_featurizer(dataset)
    mutate(dataset)
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == [dataset.name, dataset.name]
    assert feature_port._rust_featurizer_build_count(dataset) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda dataset: dataset.signatures.__setitem__("s1", object()),
        lambda dataset: dataset.papers.__setitem__("p1", object()),
        lambda dataset: setattr(dataset, "specter_embeddings", {"p1": object()}),
    ],
)
def test_unconsumed_python_state_does_not_rebuild(mutate: Any) -> None:
    dataset = DummyDataset("unconsumed")

    first = feature_port._get_rust_featurizer(dataset)
    mutate(dataset)

    assert feature_port._get_rust_featurizer(dataset) is first
    assert DummyRustFeaturizer.created == [dataset.name]


def test_exact_seed_contents_update_in_place_and_survive_rebuild() -> None:
    dataset = DummyDataset("seed-content", seed_source="python")
    dataset.cluster_seeds_require = {"s1": "c1", "s2": "c2"}
    dataset.cluster_seeds_disallow = {("s1", "s2")}

    first = feature_port._get_rust_featurizer(dataset)
    assert first.update_payloads == [({"s1": "c1", "s2": "c2"}, {("s1", "s2")})]
    assert feature_port._get_rust_featurizer(dataset) is first
    assert first.update_attempts == 1

    dataset.cluster_seeds_require["s1"] = "replacement"
    dataset.cluster_seeds_disallow.remove(("s1", "s2"))
    dataset.cluster_seeds_disallow.add(("s2", "s3"))

    assert feature_port._get_rust_featurizer(dataset) is first
    assert first.update_attempts == 2
    assert first.require == {"s1": "replacement", "s2": "c2"}
    assert first.disallow == {("s2", "s3")}
    assert feature_port._rust_featurizer_build_count(dataset) == 1

    assert feature_port.evict_rust_featurizer(dataset)
    rebuilt = feature_port._get_rust_featurizer(dataset)
    assert rebuilt is not first
    assert rebuilt.require == first.require
    assert rebuilt.disallow == first.disallow


def test_same_length_seed_container_replacement_updates_in_place() -> None:
    dataset = DummyDataset("seed-replacement", seed_source="python")
    dataset.cluster_seeds_require = {"s1": "c1", "s2": "c2"}
    dataset.cluster_seeds_disallow = {("s1", "s2")}
    featurizer = feature_port._get_rust_featurizer(dataset)

    dataset.cluster_seeds_require = {"s1": "replacement", "s2": "c2"}
    dataset.cluster_seeds_disallow = {("s2", "s3")}
    old_stamp = feature_port._rust_featurizer_state(dataset).synced_seed_stamp

    assert feature_port._get_rust_featurizer(dataset) is featurizer
    assert featurizer.update_attempts == 2
    assert featurizer.require == {"s1": "replacement", "s2": "c2"}
    assert featurizer.disallow == {("s2", "s3")}
    assert isinstance(dataset.cluster_seeds_require, feature_port._MutationTrackedDict)
    assert isinstance(dataset.cluster_seeds_disallow, feature_port._MutationTrackedSet)
    new_stamp = feature_port._rust_featurizer_state(dataset).synced_seed_stamp
    assert old_stamp is not None
    assert new_stamp is not None
    assert new_stamp[0] is not old_stamp[0]
    assert new_stamp[2] is not old_stamp[2]


def test_failed_seed_update_is_retried() -> None:
    dataset = DummyDataset("seed-retry")
    featurizer = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require = {"s1": "c1"}
    dataset.cluster_seeds_disallow = {("s1", "s2")}
    featurizer.fail_updates = True

    with pytest.raises(RuntimeError, match="seed update failed"):
        feature_port._get_rust_featurizer(dataset)

    featurizer.fail_updates = False
    assert feature_port._get_rust_featurizer(dataset) is featurizer
    assert featurizer.update_attempts == 2
    assert featurizer.require == {"s1": "c1"}
    assert featurizer.disallow == {("s1", "s2")}


def test_arrow_seed_authority_survives_rebuild_until_python_replacement() -> None:
    dataset = DummyDataset("arrow-authority")
    dataset.arrow_seed_require = {"arrow": "component"}
    dataset.arrow_seed_disallow = {("arrow", "other")}
    assert type(dataset.cluster_seeds_require) is dict
    assert type(dataset.cluster_seeds_disallow) is set

    first = feature_port._get_rust_featurizer(dataset)
    assert first.require == dataset.arrow_seed_require
    assert first.disallow == dataset.arrow_seed_disallow
    assert first.update_attempts == 0
    assert type(dataset.cluster_seeds_require) is dict
    assert type(dataset.cluster_seeds_disallow) is set

    assert feature_port.evict_rust_featurizer(dataset)
    rebuilt = feature_port._get_rust_featurizer(dataset)
    assert rebuilt.require == dataset.arrow_seed_require
    assert rebuilt.disallow == dataset.arrow_seed_disallow
    assert rebuilt.update_attempts == 0

    dataset.cluster_seeds_require = {}
    dataset.cluster_seeds_disallow = set()
    assert feature_port._get_rust_featurizer(dataset) is rebuilt
    assert dataset._cluster_seeds_source == "python"
    assert rebuilt.require == {}
    assert rebuilt.disallow == set()
    assert rebuilt.update_attempts == 1
    assert isinstance(dataset.cluster_seeds_require, feature_port._MutationTrackedDict)
    assert isinstance(dataset.cluster_seeds_disallow, feature_port._MutationTrackedSet)


def test_in_place_python_seed_before_first_get_takes_authority() -> None:
    dataset = DummyDataset("python-before-build")
    dataset.arrow_seed_require = {"arrow": "component"}
    dataset.arrow_seed_disallow = {("arrow", "other")}
    dataset.cluster_seeds_require["python"] = "component"
    dataset.cluster_seeds_disallow.add(("python", "other"))
    assert type(dataset.cluster_seeds_require) is dict
    assert type(dataset.cluster_seeds_disallow) is set

    featurizer = feature_port._get_rust_featurizer(dataset)

    assert dataset._cluster_seeds_source == "python"
    assert featurizer.require == {"python": "component"}
    assert featurizer.disallow == {("python", "other")}
    assert featurizer.update_attempts == 1
    assert isinstance(dataset.cluster_seeds_require, feature_port._MutationTrackedDict)
    assert isinstance(dataset.cluster_seeds_disallow, feature_port._MutationTrackedSet)


def test_explicit_empty_python_seeds_before_first_get_clear_arrow_seeds() -> None:
    dataset = DummyDataset("empty-python-before-build")
    dataset.arrow_seed_require = {"arrow": "component"}
    dataset.arrow_seed_disallow = {("arrow", "other")}
    dataset.cluster_seeds_require = {}
    dataset.cluster_seeds_disallow = set()

    featurizer = feature_port._get_rust_featurizer(dataset)

    assert dataset._cluster_seeds_source == "python"
    assert featurizer.require == {}
    assert featurizer.disallow == set()
    assert featurizer.update_attempts == 1


def test_inputs_changing_during_build_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = DummyDataset("changing-build-input")

    def changing_build(
        dataset_arg: DummyDataset,
    ) -> tuple[DummyRustFeaturizer, dict[str, float]]:
        dataset_arg.arrow_artifact_generation = "changed-during-build"
        return _dummy_build_rust_featurizer(dataset_arg)

    monkeypatch.setattr(feature_port, "build_rust_featurizer", changing_build)

    with pytest.raises(RuntimeError, match="inputs changed while it was being built"):
        feature_port._get_rust_featurizer(dataset)

    assert feature_port._rust_featurizer_build_count(dataset) == 0
    assert feature_port._rust_featurizer_state(dataset).featurizer is None


def test_concurrent_gets_for_same_dataset_build_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = DummyDataset("same-dataset")
    build_started = threading.Event()
    release_build = threading.Event()
    build_calls = 0

    def blocking_build(
        dataset_arg: DummyDataset,
    ) -> tuple[DummyRustFeaturizer, dict[str, float]]:
        nonlocal build_calls
        build_calls += 1
        build_started.set()
        assert release_build.wait(timeout=2)
        return _dummy_build_rust_featurizer(dataset_arg)

    monkeypatch.setattr(feature_port, "build_rust_featurizer", blocking_build)
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


def test_concurrent_seed_cache_hits_sync_one_mutation_once() -> None:
    dataset = DummyDataset("concurrent-seed-sync", seed_source="python")
    featurizer = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "c1"
    update_started = threading.Event()
    release_update = threading.Event()
    update_calls = 0
    original_update = featurizer.update_cluster_seeds

    def blocking_update(require: dict[Any, Any], disallow: set[tuple[Any, Any]]) -> None:
        nonlocal update_calls
        update_calls += 1
        update_started.set()
        assert release_update.wait(timeout=2)
        original_update(require, disallow)

    featurizer.update_cluster_seeds = blocking_update
    results: list[DummyRustFeaturizer] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            results.append(feature_port._get_rust_featurizer(dataset))
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    threads[0].start()
    assert update_started.wait(timeout=2)
    threads[1].start()
    time.sleep(0.05)
    assert update_calls == 1
    release_update.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert results == [featurizer, featurizer]
    assert update_calls == 1
    assert featurizer.require == {"s1": "c1"}


def test_distinct_datasets_build_without_a_global_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = [DummyDataset("parallel-first"), DummyDataset("parallel-second")]
    builds_ready = threading.Barrier(2)
    errors: list[Exception] = []

    def synchronized_build(
        dataset: DummyDataset,
    ) -> tuple[DummyRustFeaturizer, dict[str, float]]:
        builds_ready.wait(timeout=2)
        return _dummy_build_rust_featurizer(dataset)

    def worker(dataset: DummyDataset) -> None:
        try:
            feature_port._get_rust_featurizer(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    monkeypatch.setattr(
        feature_port,
        "build_rust_featurizer",
        synchronized_build,
    )
    threads = [threading.Thread(target=worker, args=(dataset,)) for dataset in datasets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert DummyRustFeaturizer.created == ["parallel-second", "parallel-first"]


def test_200k_seed_warm_hits_are_constant_space() -> None:
    dataset = DummyDataset("large-seed-cache", seed_source="python")
    dataset.cluster_seeds_require = dict.fromkeys(range(200_000), "component")
    featurizer = feature_port._get_rust_featurizer(dataset)

    tracemalloc.start()
    started = time.perf_counter()
    try:
        for _ in range(500):
            assert feature_port._get_rust_featurizer(dataset) is featurizer
        elapsed_seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert featurizer.update_attempts == 1
    assert elapsed_seconds < 1.0
    assert peak_bytes < 1_000_000
