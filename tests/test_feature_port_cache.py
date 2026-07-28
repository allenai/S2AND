import threading
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

import pytest

import s2and
import s2and.feature_port as feature_port
from s2and.data import ANDData


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
    __version__ = s2and.__version__
    RustFeaturizer = DummyRustFeaturizer


def _dummy_arrow_dataset() -> SimpleNamespace:
    return SimpleNamespace(native=object())


class DummyDataset(ANDData):
    def __init__(self, name: str) -> None:
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
        self.arrow_dataset = _dummy_arrow_dataset()


def _dummy_build_rust_featurizer(
    dataset: DummyDataset,
) -> tuple[DummyRustFeaturizer, dict[str, float]]:
    DummyRustFeaturizer.created.append(dataset.name)
    return (
        DummyRustFeaturizer(dataset.name),
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


def _assert_mutators_advance_version(value: Any, mutators: tuple[Any, ...]) -> None:
    for index, mutate in enumerate(mutators):
        candidate = type(value)(value)
        initial_version = candidate.mutation_version
        mutate(candidate)
        assert candidate.mutation_version > initial_version, index
        assert not hasattr(candidate, "__dict__")


def _assert_mutation_tracker_coverage() -> None:
    _assert_mutators_advance_version(
        feature_port._MutationTrackedDict({"a": 1, "b": 2}),
        (
            lambda value: value.__init__({"replacement": 3}),
            lambda value: value.__setitem__("a", 3),
            lambda value: value.__delitem__("a"),
            lambda value: value.__ior__({"replacement": 3}),
            lambda value: value.clear(),
            lambda value: value.pop("a"),
            lambda value: value.popitem(),
            lambda value: value.setdefault("replacement", 3),
            lambda value: value.update({"replacement": 3}),
        ),
    )
    _assert_mutators_advance_version(
        feature_port._MutationTrackedSet({("a", "b"), ("c", "d")}),
        (
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
        ),
    )

    def broken_pairs():
        yield ("added", 3)
        yield ("broken",)

    tracked_dict = feature_port._MutationTrackedDict({"a": 1})
    dict_version = tracked_dict.mutation_version
    with pytest.raises(ValueError):
        tracked_dict.update(broken_pairs())
    assert tracked_dict["added"] == 3
    assert tracked_dict.mutation_version > dict_version

    def broken_values():
        yield ("added", "pair")
        raise RuntimeError("broken input")

    tracked_set = feature_port._MutationTrackedSet({("a", "b")})
    set_version = tracked_set.mutation_version
    with pytest.raises(RuntimeError, match="broken input"):
        tracked_set.update(broken_values())
    assert ("added", "pair") in tracked_set
    assert tracked_set.mutation_version > set_version


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


def test_missing_mandatory_build_input_fails_fast() -> None:
    for attribute in ("arrow_dataset", "name_tuples", "preprocess", "use_orcid_id", "n_jobs"):
        dataset = DummyDataset("missing-build-input")
        delattr(dataset, attribute)

        with pytest.raises(AttributeError, match=attribute):
            feature_port._rust_featurizer_build_inputs(dataset)


def test_material_build_input_changes_rebuild() -> None:
    cases = (
        ("preprocess", lambda dataset: setattr(dataset, "preprocess", False)),
        ("use-orcid-id", lambda dataset: setattr(dataset, "use_orcid_id", False)),
        ("n-jobs", lambda dataset: setattr(dataset, "n_jobs", 2)),
        ("name-tuples", lambda dataset: dataset.name_tuples.add(("bill", "william"))),
        ("arrow-dataset", lambda dataset: setattr(dataset, "arrow_dataset", _dummy_arrow_dataset())),
    )
    for case_id, mutate in cases:
        DummyRustFeaturizer.created = []
        dataset = DummyDataset("material-input")

        first = feature_port._get_rust_featurizer(dataset)
        mutate(dataset)
        second = feature_port._get_rust_featurizer(dataset)

        assert second is not first, case_id
        assert DummyRustFeaturizer.created == [dataset.name, dataset.name], case_id
        assert feature_port._rust_featurizer_build_count(dataset) == 2, case_id


def test_unconsumed_python_state_does_not_rebuild() -> None:
    cases = (
        ("signatures", lambda dataset: dataset.signatures.__setitem__("s1", object())),
        ("papers", lambda dataset: dataset.papers.__setitem__("p1", object())),
        ("specter-embeddings", lambda dataset: setattr(dataset, "specter_embeddings", {"p1": object()})),
    )
    for case_id, mutate in cases:
        DummyRustFeaturizer.created = []
        dataset = DummyDataset("unconsumed")

        first = feature_port._get_rust_featurizer(dataset)
        mutate(dataset)

        assert feature_port._get_rust_featurizer(dataset) is first, case_id
        assert DummyRustFeaturizer.created == [dataset.name], case_id


def test_exact_seed_contents_update_in_place_and_survive_rebuild() -> None:
    _assert_mutation_tracker_coverage()
    dataset = DummyDataset("seed-content")
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
    dataset = DummyDataset("seed-replacement")
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
    initial_update_attempts = featurizer.update_attempts
    dataset.cluster_seeds_require = {"s1": "c1"}
    dataset.cluster_seeds_disallow = {("s1", "s2")}
    featurizer.fail_updates = True

    with pytest.raises(RuntimeError, match="seed update failed"):
        feature_port._get_rust_featurizer(dataset)

    featurizer.fail_updates = False
    assert feature_port._get_rust_featurizer(dataset) is featurizer
    assert featurizer.update_attempts == initial_update_attempts + 2
    assert featurizer.require == {"s1": "c1"}
    assert featurizer.disallow == {("s1", "s2")}


def test_inputs_changing_during_build_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = DummyDataset("changing-build-input")

    def changing_build(
        dataset_arg: DummyDataset,
    ) -> tuple[DummyRustFeaturizer, dict[str, float]]:
        dataset_arg.arrow_dataset = _dummy_arrow_dataset()
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
    dataset = DummyDataset("concurrent-seed-sync")
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
    dataset = DummyDataset("large-seed-cache")
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
