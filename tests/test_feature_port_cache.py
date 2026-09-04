import threading
import time
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
        self.backing = object()
        self.fail_updates = False

    def with_cluster_seeds(self, require, disallow):
        if self.fail_updates:
            raise RuntimeError("seed overlay failed")
        result = DummyRustFeaturizer(self.dataset_name, require, disallow)
        result.backing = self.backing
        return result


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


def test_cache_is_owned_by_each_dataset_and_evicts() -> None:
    first_dataset = DummyDataset("first")
    second_dataset = DummyDataset("second")

    first = feature_port._get_rust_featurizer(first_dataset)
    second = feature_port._get_rust_featurizer(second_dataset)

    assert feature_port._get_rust_featurizer(first_dataset).backing is first.backing
    assert feature_port._get_rust_featurizer(second_dataset).backing is second.backing
    assert DummyRustFeaturizer.created == ["first", "second"]
    assert feature_port._rust_featurizer_build_count(first_dataset) == 1
    assert feature_port._rust_featurizer_build_count(second_dataset) == 1

    assert feature_port.evict_rust_featurizer(first_dataset)
    assert not feature_port.evict_rust_featurizer(first_dataset)
    assert feature_port._rust_featurizer_build_count(first_dataset) == 0
    assert feature_port._get_rust_featurizer(first_dataset) is not first
    assert feature_port._get_rust_featurizer(second_dataset).backing is second.backing


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

        assert feature_port._get_rust_featurizer(dataset).backing is first.backing, case_id
        assert DummyRustFeaturizer.created == [dataset.name], case_id


def test_request_seeds_are_independent_and_do_not_replace_dataset_collections() -> None:
    dataset = DummyDataset("seed-content")
    require = dataset.cluster_seeds_require
    disallow = dataset.cluster_seeds_disallow
    first_state = SimpleNamespace(cluster_seeds_require={"s1": "a"}, cluster_seeds_disallow={("s1", "s2")})
    second_state = SimpleNamespace(cluster_seeds_require={"s1": "b"}, cluster_seeds_disallow=set())
    first = feature_port._get_rust_featurizer(dataset, prediction_state=first_state)
    second = feature_port._get_rust_featurizer(dataset, prediction_state=second_state)
    first_state.cluster_seeds_require["s1"] = "changed"
    assert first.require == {"s1": "a"}
    assert first.disallow == {("s1", "s2")}
    assert second.require == {"s1": "b"}
    assert second.disallow == set()
    assert first.backing is second.backing
    assert dataset.cluster_seeds_require is require
    assert dataset.cluster_seeds_disallow is disallow
    assert require == {}
    assert disallow == set()
    base = feature_port._rust_featurizer_state(dataset).featurizer
    assert base.require == {}
    assert base.disallow == set()
    assert feature_port._rust_featurizer_build_count(dataset) == 1


def test_dataset_seed_edits_leave_existing_handles_unchanged() -> None:
    dataset = DummyDataset("seed-edits")
    dataset.cluster_seeds_require = {"s1": "a"}
    first = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "b"
    second = feature_port._get_rust_featurizer(dataset)
    assert first.require == {"s1": "a"}
    assert second.require == {"s1": "b"}
    assert first.backing is second.backing
    assert feature_port.evict_rust_featurizer(dataset)
    rebuilt = feature_port._get_rust_featurizer(dataset)
    assert rebuilt.require == second.require
    assert rebuilt.backing is not second.backing


def test_failed_overlay_does_not_change_prior_request_or_cached_seeds() -> None:
    dataset = DummyDataset("seed-failure")
    dataset.cluster_seeds_require = {"s1": "a"}
    first = feature_port._get_rust_featurizer(dataset)
    base = feature_port._rust_featurizer_state(dataset).featurizer
    base.fail_updates = True
    dataset.cluster_seeds_require = {"s1": "b"}
    with pytest.raises(RuntimeError, match="seed overlay failed"):
        feature_port._get_rust_featurizer(dataset)
    assert first.require == {"s1": "a"}
    assert base.require == {}
    base.fail_updates = False
    second = feature_port._get_rust_featurizer(dataset)
    assert second.require == {"s1": "b"}
    assert first.backing is second.backing


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
    assert results[0] is not results[1]
    assert results[0].backing is results[1].backing


def test_concurrent_requests_keep_their_seeds_when_interleaved() -> None:
    dataset = DummyDataset("concurrent-seeds")
    ready = threading.Barrier(2)
    results = {}
    errors = []

    def worker(cluster):
        try:
            state = SimpleNamespace(cluster_seeds_require={"s1": cluster}, cluster_seeds_disallow=set())
            handle = feature_port._get_rust_featurizer(dataset, prediction_state=state)
            ready.wait(timeout=2)
            results[cluster] = handle.require["s1"]
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(cluster,)) for cluster in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert results == {"a": "a", "b": "b"}
    assert feature_port._rust_featurizer_build_count(dataset) == 1


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


def test_feature_only_hits_reuse_backing_without_reading_seeds() -> None:
    """Feature batches and warming never construct a seed overlay."""
    dataset = DummyDataset("feature-only")
    base = feature_port._get_rust_feature_data(dataset)
    base.fail_updates = True
    for _ in range(3):
        assert feature_port._get_rust_feature_data(dataset) is base
        feature_port.warm_rust_featurizer(dataset)
    assert feature_port._rust_featurizer_build_count(dataset) == 1
