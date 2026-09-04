import hashlib
import threading
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

import s2and.incremental_linking.production as production_module
import s2and.model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import (
    read_cluster_seed_disallows_arrow,
    read_incremental_query_signatures_arrow,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_cluster_seeds_arrow,
    write_name_counts_index,
)
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    RawArrowPlanBundle,
)
from s2and.incremental_linking.runtime import LinkOrAbstainDecision
from s2and.model import Clusterer, IncrementalDistStats
from s2and.prediction_state import PredictionState
from tests.helpers import (
    tiny_name_counts_index,
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
    write_test_arrow_artifact_manifest,
)

_PROMOTED_TEST_FEATURIZER_INFO = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
_PROMOTED_TEST_FEATURE_COLUMNS = ("test_link_probability",)


def _same_partition(a: dict[str, list[str]], b: dict[str, list[str]]) -> bool:
    """Check that two cluster dicts encode the same partition (same groupings, ignoring cluster IDs)."""

    def _to_partition(clusters: dict[str, list[str]]) -> frozenset:
        return frozenset(frozenset(sigs) for sigs in clusters.values() if sigs)

    return _to_partition(a) == _to_partition(b)


def _clusters(result: dict[str, Any]) -> dict[str, list[str]]:
    return dict(result["clusters"])


def _minimal_arrow_dataset(tmp_path: Path) -> ArrowDataset:
    write_minimal_arrow_prediction_bundle(tmp_path)
    return ArrowDataset.open(tmp_path)


def _signature_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    import pyarrow as pa

    def strings(key: str, default: str | None = "") -> Any:
        return pa.array([row.get(key, default) for row in rows], type=pa.string())

    columns = {
        "signature_id": strings("signature_id"),
        "paper_id": strings("paper_id"),
        "author_first": strings("author_first"),
        "author_middle": strings("author_middle"),
        "author_last": strings("author_last"),
        "author_suffix": strings("author_suffix"),
        "author_affiliations": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
        "author_position": pa.array([0 for _row in rows], type=pa.int64()),
    }
    if any("author_orcid" in row for row in rows):
        columns["author_orcid"] = strings("author_orcid", default=None)
    return pa.table(columns)


def _direct_arrow_dataset(
    *,
    cluster_seeds_disallow: set[tuple[str, str]] | None = None,
    signatures: Mapping[str, Any] | None = None,
) -> Any:
    return production_module._DirectArrowIncrementalDataset(
        name_tuples=set(),
        cluster_seeds_require={"seed": "c_seed"},
        cluster_seeds_disallow=cluster_seeds_disallow or set(),
        altered_cluster_signatures=None,
        max_seed_cluster_id=0,
        signatures=signatures or {},
    )


def _patch_fake_raw_arrow_planner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, Any] | None = None,
) -> None:
    """Install a planner-shaped Rust fake for promoted raw Arrow orchestration tests."""

    class FakePlanner:
        def __init__(self, resource: object, query_signature_ids: list[str], **_kwargs: object):
            self._resource = resource
            self._query_signature_ids = tuple(query_signature_ids)
            self._name_counts_index = object()
            if captured is not None:
                captured.setdefault("planner_inits", []).append(self._query_signature_ids)
                captured.setdefault("planner_refs", []).append(weakref.ref(self))
                captured["planner_name_counts_index"] = self._name_counts_index

        @classmethod
        def from_query_signatures(
            cls,
            resource: object,
            query_signatures_path: str,
            **kwargs: object,
        ) -> "FakePlanner":
            rows = read_incremental_query_signatures_arrow(Path(query_signatures_path))
            return cls(resource, [row.signature_id for row in rows], **kwargs)

        @classmethod
        def from_auto_queries(
            cls,
            resource: object,
            _cluster_seeds_path: str,
            **kwargs: object,
        ) -> "FakePlanner":
            if captured is not None:
                disallow_path = kwargs.get("cluster_seed_disallows_path")
                captured.setdefault("planner_disallows", []).append(
                    set() if disallow_path is None else set(read_cluster_seed_disallows_arrow(Path(str(disallow_path))))
                )
            return cls(resource, [], **kwargs)

        def build_telemetry(self):
            return {
                "query_signature_count": len(self._query_signature_ids),
                "signature_count": len(self._query_signature_ids),
            }

        def name_counts_index(self) -> object:
            if captured is not None:
                captured["planner_name_counts_index_calls"] = (
                    int(captured.get("planner_name_counts_index_calls", 0)) + 1
                )
            return self._name_counts_index

        def plan(
            self,
            query_signature_ids: list[str],
            additional_cluster_seed_disallows: list[tuple[str, str]] | None = None,
            **_kwargs: object,
        ):
            query_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_plans", []).append(query_ids)
                captured.setdefault("planner_plan_disallows", []).append(set(additional_cluster_seed_disallows or ()))
            plan: dict[str, Any] = {
                "query_signature_ids": query_ids,
                "query_views": ["full"] * len(query_ids),
                "query_authors": [""] * len(query_ids),
                "row_count": 0,
                "pair_count": 0,
                "row_query_signature_indices": np.asarray([], dtype=np.uint32),
                "row_component_keys": [],
                "retrieval_scores": np.asarray([], dtype=np.float32),
                "retrieval_ranks": np.asarray([], dtype=np.uint16),
                "pair_row_indices": np.asarray([], dtype=np.uint32),
                "left_signature_ids": [],
                "right_signature_ids": [],
                "component_members": {},
            }
            for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
                plan[raw_key] = np.asarray([], dtype=dtype)
            return plan

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_dataset",
        lambda *_args, **_kwargs: object(),
    )


def test_raw_plan_contiguous_query_slice_rebases_rows_and_pairs() -> None:
    plan: dict[str, Any] = {
        "query_signature_ids": ["q0", "q1", "q2"],
        "query_views": ["full", "full", "full"],
        "query_authors": ["A", "B", "C"],
        "row_count": 4,
        "pair_count": 4,
        "row_query_signature_indices": np.asarray([0, 0, 1, 2], dtype=np.uint32),
        "row_component_keys": ["c0", "c1", "c2", "c3"],
        "retrieval_scores": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "retrieval_ranks": np.asarray([1, 2, 1, 1], dtype=np.uint16),
        "pair_row_indices": np.asarray([0, 1, 2, 3], dtype=np.uint32),
        "left_signature_ids": ["q0", "q0", "q1", "q2"],
        "right_signature_ids": ["s0", "s1", "s2", "s3"],
        "component_members": {f"c{index}": [f"s{index}"] for index in range(4)},
        "telemetry": {"query_signature_count": 3},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        fill_value: Any = "" if dtype is object else 0
        plan[raw_key] = np.asarray([fill_value] * 4, dtype=dtype)
    bundle = RawArrowPlanBundle.from_native_mapping(plan)

    first = bundle.contiguous_query_slice(0, 1)
    remainder = bundle.contiguous_query_slice(1, 3)

    assert first.telemetry == bundle.telemetry
    assert remainder.telemetry is None
    assert remainder.query_signature_ids == ("q1", "q2")
    assert remainder.row_query_offsets.tolist() == [0, 1]
    assert remainder.pair_row_indices.tolist() == [0, 1]
    assert remainder.left_signature_ids == ("q1", "q2")
    assert remainder.right_signature_ids == ("s2", "s3")
    assert np.shares_memory(remainder.retrieval_scores, bundle.retrieval_scores)


def test_finish_incremental_uses_split_inverse_for_altered_incompatibility_check() -> None:
    """A split altered profile should compare new names only against the linked split."""

    def signature(first: str) -> SimpleNamespace:
        return SimpleNamespace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=first,
            author_info_last="Jones",
            paper_id=f"p-{first}",
        )

    dataset = SimpleNamespace(
        signatures={
            "seed_david": signature("David"),
            "seed_initial": signature("D"),
            "new_donald": signature("Donald"),
        },
        name_tuples=set(),
        max_seed_cluster_id=0,
    )
    clusterer = SimpleNamespace(
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
    )

    clusters = Clusterer._finish_incremental_with_seed_links(
        cast(Any, clusterer),
        ["new_donald"],
        cast(Any, dataset),
        {"new_donald": "0_1"},
        {"0_0": "0", "0_1": "0"},
        {"0": ["seed_david", "seed_initial"]},
        prevent_new_incompatibilities=True,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace()),
        split_cluster_seeds_require_inverse={
            "0_0": ["seed_david"],
            "0_1": ["seed_initial"],
        },
    )

    assert clusters == {"0": ["seed_david", "seed_initial", "new_donald"]}


def test_finish_incremental_lazily_resolves_default_name_tuples_for_direct_arrow_dataset() -> None:
    """Direct Arrow's default name_tuples value should not crash compatibility checks."""

    def signature(first: str) -> SimpleNamespace:
        normalized = first.lower()
        return SimpleNamespace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=normalized,
            author_info_last="Jones",
            paper_id=f"p-{normalized}",
        )

    dataset = SimpleNamespace(
        signatures={
            "seed_xavier": signature("Xavier"),
            "new_zelda": signature("Zelda"),
        },
        name_tuples=None,
        max_seed_cluster_id=0,
    )
    clusterer = SimpleNamespace(
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
    )

    clusters = Clusterer._finish_incremental_with_seed_links(
        cast(Any, clusterer),
        ["new_zelda"],
        cast(Any, dataset),
        {"new_zelda": "0_0"},
        {"0_0": "0"},
        {"0": ["seed_xavier"]},
        prevent_new_incompatibilities=True,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace()),
        split_cluster_seeds_require_inverse={"0_0": ["seed_xavier"]},
    )

    assert clusters == {"0": ["seed_xavier"], "1": ["new_zelda"]}


def test_finish_incremental_resolves_name_tuples_once_for_all_assignments(monkeypatch: pytest.MonkeyPatch) -> None:
    def signature(first: str) -> SimpleNamespace:
        return SimpleNamespace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=first.lower(),
            author_info_last="Jones",
            paper_id=f"p-{first.lower()}",
        )

    dataset = SimpleNamespace(
        signatures={
            "seed": signature("William"),
            "new_1": signature("Bill"),
            "new_2": signature("Bill"),
            "new_3": signature("Bill"),
        },
        name_tuples={("william", "bill")},
        max_seed_cluster_id=0,
    )
    clusterer = SimpleNamespace(
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
    )
    original_resolver = model_module._name_tuples_for_incremental_rules
    resolver_calls = 0

    def recording_resolver(name_tuples):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(name_tuples)

    monkeypatch.setattr(model_module, "_name_tuples_for_incremental_rules", recording_resolver)
    clusters = Clusterer._finish_incremental_with_seed_links(
        cast(Any, clusterer),
        ["new_1", "new_2", "new_3"],
        cast(Any, dataset),
        {"new_1": "0_0", "new_2": "0_1", "new_3": "0_2"},
        {"0_0": "0", "0_1": "0", "0_2": "0"},
        {"0": ["seed"]},
        prevent_new_incompatibilities=True,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace()),
        split_cluster_seeds_require_inverse={
            "0_0": ["seed"],
            "0_1": ["seed"],
            "0_2": ["seed"],
        },
    )

    assert clusters == {"0": ["seed", "new_1", "new_2", "new_3"]}
    assert resolver_calls == 1


def test_model_presplit_cache_fingerprint_drops_cluster_model_identity() -> None:
    class DummyClusterModel:
        def get_params(self, *, deep: bool = False) -> dict[str, float]:
            return {"eps": 0.5}

    classifier = object()
    nameless_classifier = object()
    base = {
        "classifier": classifier,
        "nameless_classifier": nameless_classifier,
        "featurizer_info": SimpleNamespace(features_to_use=("year_diff",)),
        "nameless_featurizer_info": SimpleNamespace(features_to_use=()),
        "use_default_constraints_as_supervision": True,
        "dont_merge_cluster_seeds": True,
        "suppress_orcid": False,
    }
    first = SimpleNamespace(**base, cluster_model=DummyClusterModel())
    second = SimpleNamespace(**base, cluster_model=DummyClusterModel())

    assert model_module._model_presplit_cache_fingerprint(first) == model_module._model_presplit_cache_fingerprint(
        second
    )


def test_model_presplit_cache_fingerprint_tracks_classifier_state() -> None:
    classifier = SimpleNamespace(version=1)
    clusterer = SimpleNamespace(
        classifier=classifier,
        nameless_classifier=None,
        cluster_model=None,
        featurizer_info=SimpleNamespace(features_to_use=("year_diff",)),
        nameless_featurizer_info=SimpleNamespace(features_to_use=()),
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        suppress_orcid=False,
    )

    before = model_module._model_presplit_cache_fingerprint(clusterer)
    classifier.version = 2

    assert model_module._model_presplit_cache_fingerprint(clusterer) != before


def test_model_presplit_cache_fingerprint_hashes_complete_numpy_fitted_state() -> None:
    fit_x = np.concatenate((np.zeros((1_000, 1)), np.full((1_000, 1), 10.0)))
    fit_y = np.concatenate((np.zeros(1_000, dtype=int), np.ones(1_000, dtype=int)))
    classifier = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(fit_x, fit_y)
    clusterer = SimpleNamespace(
        classifier=classifier,
        nameless_classifier=None,
        cluster_model=None,
        featurizer_info=SimpleNamespace(features_to_use=("year_diff",)),
        nameless_featurizer_info=SimpleNamespace(features_to_use=()),
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        suppress_orcid=False,
    )

    legacy_state_repr = repr(classifier.__dict__)
    before = model_module._model_presplit_cache_fingerprint(clusterer)
    assert model_module._model_presplit_cache_fingerprint(clusterer) == before
    assert classifier.predict([[9.0]]).item() == 1

    # This fitted row is hidden behind NumPy's repr ellipsis but changes predictions.
    classifier._fit_X[500, 0] = 9.0

    assert repr(classifier.__dict__) == legacy_state_repr
    assert classifier.predict([[9.0]]).item() == 0
    assert model_module._model_presplit_cache_fingerprint(clusterer) != before


def test_estimator_cache_fingerprint_disables_reuse_without_complete_serialization() -> None:
    estimator = SimpleNamespace(unpickleable=lambda: None)

    assert model_module._estimator_cache_fingerprint(estimator) != model_module._estimator_cache_fingerprint(estimator)


def test_altered_presplit_lru_serializes_concurrent_get_and_eviction() -> None:
    read_paused = threading.Event()
    allow_read_return = threading.Event()
    writer_done = threading.Event()

    class PausingCache(OrderedDict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if key == ("target",):
                read_paused.set()
                assert allow_read_return.wait(timeout=5)
            return value

    target_key = ("target",)
    cache = PausingCache(
        [(target_key, (("target-sig",),))] + [((f"filler-{index}",), ((f"sig-{index}",),)) for index in range(127)]
    )
    clusterer = SimpleNamespace(_s2and_altered_presplit_cache=cache)
    reader_values: list[tuple[tuple[str, ...], ...] | None] = []
    thread_errors: list[Exception] = []

    def read_target() -> None:
        try:
            reader_values.append(model_module._get_altered_presplit_cache_entry(clusterer, target_key))
        except Exception as exc:  # pragma: no cover - asserted below
            thread_errors.append(exc)

    def evict_oldest() -> None:
        try:
            model_module._put_altered_presplit_cache_entry(clusterer, ("new",), [["new-sig"]])
        except Exception as exc:  # pragma: no cover - asserted below
            thread_errors.append(exc)
        finally:
            writer_done.set()

    reader = threading.Thread(target=read_target)
    writer = threading.Thread(target=evict_oldest)
    reader.start()
    assert read_paused.wait(timeout=5)
    writer.start()
    writer_finished_during_get = writer_done.wait(timeout=1)
    allow_read_return.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert not writer_finished_during_get
    assert thread_errors == []
    assert reader_values == [(("target-sig",),)]
    assert target_key in cache


def test_cluster_seed_inverse_canonicalizes_component_ids() -> None:
    assert model_module._cluster_seeds_require_inverse({"s1": 7, "s2": "7"}) == {"7": ["s1", "s2"]}


def test_predict_from_rust_featurizer_skips_unused_signature_rule_metadata() -> None:
    captured: dict[str, Any] = {}

    class DummyClusterer:
        predict_from_rust_featurizer = Clusterer.predict_from_rust_featurizer

        def predict_helper(self, block_dict, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            return {"block_0": list(block_dict["block"])}, kwargs["dists"]

    class FakeRustFeaturizer:
        def signature_rule_metadata(self):
            raise AssertionError("full-block prediction does not consume signature rule metadata")

    clusterer = DummyClusterer()
    cast(Any, clusterer).predict_from_rust_featurizer(
        {"block": ["s_alice", "s_bob", "s_alicia"]},
        FakeRustFeaturizer(),
        dists={"block": np.asarray([0.1, 0.2, 0.3], dtype=np.float64)},
        cluster_seeds_require={},
    )

    proxy_dataset = captured["dataset"]
    assert proxy_dataset.signatures == {}


def _build_dummy_clusterer_and_dataset(*, name: str = "dummy_chunked") -> tuple[Clusterer, ANDData]:
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        cluster_seeds={"6": {"7": "require"}, "3": {"4": "require"}},
        name=name,
        name_counts_index=tiny_name_counts_index(),
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(1)
    X_random = rng.random((10, 6))
    y_random = rng.randint(0, 6, 10)
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1, verbosity=-1).fit(
            X_random, y_random
        ),
        n_jobs=1,
        use_default_constraints_as_supervision=True,
    )
    return clusterer, dataset


@pytest.fixture
def clusterer_dataset_factory():
    def _factory(*, name: str = "dummy_chunked") -> tuple[Clusterer, ANDData]:
        return _build_dummy_clusterer_and_dataset(name=name)

    return _factory


@pytest.fixture(autouse=True)
def _use_python_backend_by_default(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "python")


def test_predict_incremental(clusterer_dataset_factory):
    # base clustering of the random model would be
    # {'0': ['0', '1', '2'], '1': ['3', '4', '5', '8'], '2': ['6', '7']}
    dummy_clusterer, dummy_dataset = clusterer_dataset_factory(name="dummy")
    block = ["3", "4", "5", "6", "7", "8"]

    # Non-subblocked (monolithic) is the reference output.
    output_monolithic = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset))
    expected_output = {"0": ["6", "7"], "1": ["3", "4", "5", "8"]}
    assert _same_partition(output_monolithic, expected_output)

    dummy_dataset.cluster_seeds_disallow = {("5", "7"), ("8", "4"), ("5", "4"), ("8", "7")}
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset))
    expected_output = {"0": ["6", "7"], "1": ["3", "4"], "2": ["5", "8"]}
    assert _same_partition(output, expected_output)

    dummy_dataset.altered_cluster_signatures = ["1", "5"]
    dummy_dataset.cluster_seeds_require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
    block = ["3", "4", "8"]
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset))
    expected_output = {"0": ["1", "2", "5", "8"], "1": ["6", "7", "3", "4"]}
    assert _same_partition(output, expected_output)


def test_predict_incremental_return_contract(clusterer_dataset_factory):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_contract")

    payload = clusterer.predict_incremental(block, dataset)
    assert set(payload) >= {"clusters", "phase_b_mode", "phase_b_budget_bytes", "phase_b_required_bytes"}
    assert payload["phase_b_mode"] == "exact"


@pytest.mark.parametrize("ignore_seeds, expected_count", [(False, 1), (True, 2)])
def test_predict_posthoc_seed_merge_respects_ignore_flag(ignore_seeds: bool, expected_count: int) -> None:
    """Seed postprocessing must not undo clustering when requires are disabled."""
    clusterer = Clusterer(FeaturizationInfo(features_to_use=["year_diff"]), object(), n_jobs=1)
    dataset = SimpleNamespace(cluster_seeds_require={"s1": "claimed", "s2": "claimed"}, cluster_seeds_disallow=set())
    clusters, _ = clusterer.predict(
        {"block": ["s1", "s2"]},
        dataset,
        dists={"block": np.array([1.0])},
        incremental_dont_use_cluster_seeds=ignore_seeds,
    )
    assert len(clusters) == expected_count


def test_predict_ignore_seed_requires_preserves_explicit_disallows(clusterer_dataset_factory) -> None:
    """Disabling required seed groups must retain explicit cannot-link constraints."""
    clusterer, dataset = clusterer_dataset_factory(name="ignore_seeds_disallow")
    dataset.cluster_seeds_require = {"3": "claimed", "4": "claimed"}
    dataset.cluster_seeds_disallow = set()
    control, _ = clusterer.predict({"block": ["3", "4"]}, dataset, incremental_dont_use_cluster_seeds=True)
    assert len(control) == 1
    dataset.cluster_seeds_disallow = {("3", "4")}
    clusters, _ = clusterer.predict({"block": ["3", "4"]}, dataset, incremental_dont_use_cluster_seeds=True)
    assert len(clusters) == 2


def test_predict_incremental_links_against_altered_components(clusterer_dataset_factory) -> None:
    """Link to natural altered-profile components before restoring the claimed ID."""
    clusterer, dataset = clusterer_dataset_factory(name="altered_components")
    dataset.cluster_seeds_require = {"0": "claimed", "3": "claimed"}
    dataset.cluster_seeds_disallow = set()
    dataset.altered_cluster_signatures = ["0"]
    result = _clusters(clusterer.predict_incremental(["4"], dataset))
    dataset.cluster_seeds_require = {"0": "split0", "3": "split3"}
    dataset.altered_cluster_signatures = []
    control = _clusters(clusterer.predict_incremental(["4"], dataset))
    assert control["split3"] == ["3", "4"]
    assert set(result["claimed"]) == {"0", "3", "4"}


def test_promoted_incremental_orcid_fanout_by_query_counts_matching_components() -> None:
    class CountingSeedMap(dict[str, str]):
        values_call_count = 0

        def values(self):
            self.values_call_count += 1
            return super().values()

    dataset = SimpleNamespace(
        signatures={
            "q": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "q_alias": SimpleNamespace(author_info_orcid="0000-0000-0000-0001"),
            "blank": SimpleNamespace(author_info_orcid="   "),
            "other": SimpleNamespace(author_info_orcid="0000-0000-0000-0002"),
            "seed_a": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "seed_b": SimpleNamespace(author_info_orcid="0000-0000-0000-0001"),
            "seed_b_duplicate": SimpleNamespace(author_info_orcid="0000-0000-0000-0001"),
            "seed_c": SimpleNamespace(author_info_orcid="0000-0000-0000-0003"),
            "seed_blank": SimpleNamespace(author_info_orcid="   "),
        }
    )
    seed_map = CountingSeedMap(
        {
            "seed_a": "cluster_a",
            "seed_b": "cluster_b",
            "seed_b_duplicate": "cluster_b",
            "seed_c": "cluster_b",
            "seed_blank": "cluster_blank",
        }
    )
    fanout = production_module.promoted_incremental_orcid_fanout_by_query(
        dataset,  # type: ignore[arg-type]
        ["q", "q_alias", "blank", "other"],
        seed_map,
        orcid_enabled=True,
        component_sizes={"cluster_a": 1, "cluster_b": 3, "cluster_blank": 1},
    )

    assert fanout == {"q": (2, 4), "q_alias": (2, 4)}
    assert seed_map.values_call_count == 0
    assert (
        production_module.promoted_incremental_orcid_fanout_by_query(
            dataset,  # type: ignore[arg-type]
            ["q"],
            {"seed_a": "cluster_a"},
            orcid_enabled=False,
        )
        == {}
    )


def test_predict_incremental_from_arrow_uses_promoted_linker_without_anddata(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow")
    clusterer.incremental_linker_artifact = SimpleNamespace(artifact_dir=tmp_path)
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    block = ["3", "4", "5", "6"]
    runtime_context = cast(Any, SimpleNamespace(backend="rust", run_id="test-direct-arrow"))
    captured: dict[str, Any] = {}
    payload = {
        "clusters": {"0": ["3", "4", "5"], "1": ["6"]},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    def fake_load_signature_info(_lease: Any, signature_ids: Any) -> dict[str, Any]:
        captured["metadata_signature_ids"] = tuple(signature_ids)
        return {
            str(signature_id): model_module._ArrowIncrementalSignatureInfo(
                paper_id=f"p{signature_id}",
                author_info_first="Alex",
                author_info_last="Smith",
                author_info_first_normalized_without_apostrophe="alex",
                author_info_orcid=None,
            )
            for signature_id in signature_ids
        }

    def fake_promoted_linker(clusterer_arg: Any, block_signatures_arg: list[str], dataset_arg: Any, **kwargs: Any):
        captured["clusterer"] = clusterer_arg
        captured["block_signatures"] = list(block_signatures_arg)
        captured["dataset"] = dataset_arg
        captured.update(kwargs)
        return dict(payload)

    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow",
        fake_promoted_linker,
    )

    result = clusterer.predict_incremental_from_arrow(
        block,
        arrow_dataset,
        batching_threshold=2,
        total_ram_bytes=100_000,
        runtime_context=runtime_context,
        name_tuples={("alex", "al")},
        cluster_seeds_require={"3": "0", "4": "0"},
    )

    assert result == payload
    assert captured["clusterer"] is clusterer
    assert captured["block_signatures"] == block
    assert not isinstance(captured["dataset"], ANDData)
    assert captured["dataset"].cluster_seeds_require == {"3": "0", "4": "0"}
    assert captured["dataset"].name_tuples == {("al", "alex")}
    assert captured["arrow_dataset"] is arrow_dataset
    assert captured["runtime_context"] is runtime_context
    assert captured["batching_threshold"] == 2
    assert captured["total_ram_bytes"] == 100_000
    assert captured["metadata_signature_ids"] == ("5", "6")
    assert set(captured["dataset"].signatures) == {"5", "6"}


def test_predict_incremental_from_arrow_requires_explicit_seeds(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_no_seeds")
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("promoted linker should not run")),
    )

    with pytest.raises(ValueError, match="requires nonempty cluster_seeds_require"):
        clusterer.predict_incremental_from_arrow(["3", "4", "5"], arrow_dataset)


def test_predict_incremental_from_arrow_loads_altered_seed_metadata(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_altered")
    clusterer.incremental_linker_artifact = SimpleNamespace(artifact_dir=tmp_path)
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_signature_info(_lease: Any, signature_ids: Any) -> dict[str, Any]:
        captured["metadata_signature_ids"] = tuple(signature_ids)
        return {
            str(signature_id): model_module._ArrowIncrementalSignatureInfo(
                paper_id=f"p{signature_id}",
                author_info_first="Alex",
                author_info_last="Smith",
                author_info_first_normalized_without_apostrophe="alex",
                author_info_orcid=None,
            )
            for signature_id in signature_ids
        }

    def fake_promoted_linker(_clusterer: Any, _block_signatures: list[str], dataset_arg: Any, **_kwargs: Any):
        captured["dataset"] = dataset_arg
        return {"clusters": {"0": ["3", "4", "5"]}}

    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow",
        fake_promoted_linker,
    )

    clusterer.predict_incremental_from_arrow(
        ["3", "4", "5"],
        arrow_dataset,
        cluster_seeds_require={"3": "0", "4": "0", "8": "1"},
        altered_cluster_signatures=["3"],
    )

    assert set(captured["metadata_signature_ids"]) == {"3", "4", "5"}
    assert captured["dataset"].altered_cluster_signatures == ["3"]
    assert set(captured["dataset"].signatures) == {"3", "4", "5"}


def test_predict_incremental_from_arrow_loads_seed_orcids_for_budget_floor(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_orcid_floor")
    clusterer.incremental_linker_artifact = SimpleNamespace(artifact_dir=tmp_path)
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_signature_info(_lease: Any, signature_ids: Any) -> dict[str, Any]:
        captured["metadata_signature_ids"] = tuple(signature_ids)
        return {
            str(signature_id): model_module._ArrowIncrementalSignatureInfo(
                paper_id=f"p{signature_id}",
                author_info_first="Alex",
                author_info_last="Smith",
                author_info_first_normalized_without_apostrophe="alex",
                author_info_orcid="0000-0000-0000-0001" if str(signature_id) == "5" else None,
            )
            for signature_id in signature_ids
        }

    def fake_load_orcid_info(_lease: Any, signature_ids: Any) -> dict[str, Any]:
        captured["orcid_signature_ids"] = tuple(signature_ids)
        return {
            str(signature_id): model_module._ArrowIncrementalSignatureInfo(
                paper_id=None,
                author_info_first=None,
                author_info_last=None,
                author_info_first_normalized_without_apostrophe=None,
                author_info_orcid="0000-0000-0000-0001" if str(signature_id) == "8" else None,
            )
            for signature_id in signature_ids
        }

    def fake_promoted_linker(_clusterer: Any, _block_signatures: list[str], dataset_arg: Any, **_kwargs: Any):
        captured["fanout"] = production_module.promoted_incremental_orcid_fanout_by_query(
            dataset_arg,
            ["5"],
            dataset_arg.cluster_seeds_require,
            orcid_enabled=True,
        )
        return {"clusters": {"0": ["3", "4"], "1": ["8", "5"]}}

    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_orcid_signature_info", fake_load_orcid_info)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow",
        fake_promoted_linker,
    )

    clusterer.predict_incremental_from_arrow(
        ["3", "4", "5", "8"],
        arrow_dataset,
        cluster_seeds_require={"3": "0", "4": "0", "8": "1"},
    )

    assert captured["metadata_signature_ids"] == ("5",)
    assert captured["orcid_signature_ids"] == ("3", "4", "8")
    assert captured["fanout"] == {"5": (1, 1)}


def test_load_arrow_incremental_signature_info_uses_signature_batch_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    signatures_path = Path(arrow_paths["signatures"])
    batches = [
        _signature_table(
            [
                {
                    "signature_id": "s1",
                    "paper_id": "p1",
                    "author_first": "Alice",
                    "author_last": "Jones",
                    "author_orcid": None,
                }
            ]
        ),
        _signature_table(
            [
                {
                    "signature_id": "s2",
                    "paper_id": "p2",
                    "author_first": "Anne-Marie",
                    "author_last": "Ng",
                    "author_orcid": "0000-0002",
                },
                {
                    "signature_id": "s3",
                    "paper_id": None,
                    "author_first": "Null",
                    "author_last": "Paper",
                    "author_orcid": None,
                },
            ]
        ),
    ]
    with pa.OSFile(str(signatures_path), "wb") as sink:
        with pa.ipc.new_file(sink, batches[0].schema) as writer:
            for batch in batches:
                writer.write_table(batch)
    index_path = Path(arrow_paths["signatures_batch_index"])
    write_arrow_batch_lookup_index(
        signatures_path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
    )

    original_open_file = pa.ipc.open_file
    read_batch_indices: list[int] = []

    class CountingReader:
        def __init__(self, reader: Any) -> None:
            self._reader = reader

        def __getattr__(self, name: str) -> Any:
            return getattr(self._reader, name)

        def get_batch(self, index: int) -> Any:
            read_batch_indices.append(index)
            return self._reader.get_batch(index)

    def counting_open_file(source: Any) -> CountingReader:
        return CountingReader(original_open_file(source))

    monkeypatch.setattr(pa.ipc, "open_file", counting_open_file)

    write_test_arrow_artifact_manifest(tmp_path, arrow_paths)
    arrow_dataset = ArrowDataset.open(tmp_path)
    with arrow_dataset.use() as lease:
        signatures = model_module._load_arrow_incremental_signature_info(lease, ["s2", "s3"])

        assert set(signatures) == {"s2", "s3"}
        assert signatures["s2"].paper_id == "p2"
        assert signatures["s2"].author_info_first == "Anne-Marie"
        assert signatures["s2"].author_info_last == "Ng"
        assert signatures["s2"].author_info_orcid == "0000-0002"
        assert signatures["s3"].paper_id is None
        assert read_batch_indices == [1]

        read_batch_indices.clear()
        orcid_signatures = model_module._load_arrow_incremental_orcid_signature_info(lease, ["s2"])
        assert set(orcid_signatures) == {"s2"}
        assert orcid_signatures["s2"].paper_id is None
        assert orcid_signatures["s2"].author_info_first is None
        assert orcid_signatures["s2"].author_info_orcid == "0000-0002"
        assert read_batch_indices == [1]


def test_load_arrow_incremental_orcid_signature_info_skips_missing_seed_ids(tmp_path: Path) -> None:
    import pyarrow as pa

    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    signatures_path = Path(arrow_paths["signatures"])
    table = _signature_table(
        [
            {
                "signature_id": "query",
                "paper_id": "p1",
                "author_orcid": "0000-0000-0000-0001",
            }
        ]
    )
    with pa.OSFile(str(signatures_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    index_path = Path(arrow_paths["signatures_batch_index"])
    write_arrow_batch_lookup_index(
        signatures_path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
    )

    write_test_arrow_artifact_manifest(tmp_path, arrow_paths)
    arrow_dataset = ArrowDataset.open(tmp_path)
    with arrow_dataset.use() as lease:
        assert model_module._load_arrow_incremental_orcid_signature_info(lease, ["stale_seed"]) == {}


def test_load_arrow_incremental_orcid_signature_info_accepts_missing_orcid_column(tmp_path: Path) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    signatures_path = Path(arrow_paths["signatures"])
    write_arrow_ipc_table(
        _signature_table([{"signature_id": "query", "paper_id": "p1"}]),
        signatures_path,
    )
    index_path = Path(arrow_paths["signatures_batch_index"])
    write_arrow_batch_lookup_index(
        signatures_path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
    )

    write_test_arrow_artifact_manifest(tmp_path, arrow_paths)
    arrow_dataset = ArrowDataset.open(tmp_path)
    with arrow_dataset.use() as lease:
        assert model_module._load_arrow_incremental_orcid_signature_info(lease, ["query"]) == {}


def test_direct_arrow_incremental_requires_loaded_artifact() -> None:
    with pytest.raises(FileNotFoundError, match="requires an attached incremental linker artifact"):
        model_module._required_incremental_linker_artifact(SimpleNamespace())

    attached = SimpleNamespace(artifact_dir=Path("explicit-linker"))
    assert (
        model_module._required_incremental_linker_artifact(SimpleNamespace(incremental_linker_artifact=attached))
        is attached
    )


def test_partial_supervision_plan_disallows_keep_only_explicit_query_to_active_seed_pairs() -> None:
    partial_supervision = {
        ("seed-1", "query-1"): LARGE_DISTANCE,
        ("query-1", "seed-1"): LARGE_DISTANCE,
        ("query-2", "seed-2"): float(LARGE_DISTANCE),
        ("query-1", "query-2"): LARGE_DISTANCE,
        ("seed-1", "seed-2"): LARGE_DISTANCE,
        ("query-1", "unknown"): LARGE_DISTANCE,
        ("unknown", "seed-1"): LARGE_DISTANCE,
        ("query-1", "seed-2"): LARGE_DISTANCE - 1,
    }

    result = production_module._partial_supervision_plan_disallows(  # noqa: SLF001
        partial_supervision,
        query_signature_ids=["query-1", "query-2"],
        seed_signature_ids=["seed-1", "seed-2"],
    )

    assert result == {
        ("query-1", "seed-1"),
        ("query-2", "seed-2"),
    }


def test_promoted_linker_adds_partial_query_seed_disallows_to_planner_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SidecarCaptured(Exception):
        pass

    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    captured: dict[str, Any] = {}

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **kwargs: object):
            captured["request_disallows"] = set(kwargs["cluster_seed_disallows"])
            kwargs["prediction_state"].telemetry["incremental_seed_setup"] = {
                "seed_setup_cluster_seeds_source": "python"
            }
            seeds = {"seed-1": "component-1", "seed-2": "component-2"}
            inverse = {"component-1": ["seed-1"], "component-2": ["seed-2"]}
            return seeds, {}, inverse, inverse

    @contextmanager
    def capture_sidecar(
        seeds: Mapping[str, str],
        *,
        prefix: str,
        cluster_seeds_disallow: set[tuple[str, str]],
    ):
        captured["seeds"] = dict(seeds)
        captured["prefix"] = prefix
        captured["planner_disallows"] = set(cluster_seeds_disallow)
        raise SidecarCaptured
        yield

    monkeypatch.setattr(
        production_module,
        "temporary_cluster_seed_sidecars",
        capture_sidecar,
    )
    dataset = _direct_arrow_dataset(cluster_seeds_disallow={("query", "seed-1")})
    partial_supervision = {("seed-2", "query"): LARGE_DISTANCE}

    with pytest.raises(SidecarCaptured):
        production_module.predict_incremental_promoted_linker_from_arrow(
            FakeClusterer(),
            ["seed-1", "seed-2", "query"],
            dataset,
            arrow_dataset=_minimal_arrow_dataset(tmp_path),
            artifact=FakeArtifact(),
            prevent_new_incompatibilities=False,
            partial_supervision=partial_supervision,
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
        )

    assert captured["request_disallows"] == {("query", "seed-1")}
    assert captured["planner_disallows"] == {
        ("query", "seed-1"),
        ("query", "seed-2"),
    }
    assert dataset.cluster_seeds_disallow == {("query", "seed-1")}
    assert partial_supervision == {("seed-2", "query"): LARGE_DISTANCE}


def test_predict_incremental_arrow_promoted_linker_cleans_up_temp_seed_context_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []

    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    @contextmanager
    def fake_temporary_cluster_seed_sidecars(*_args: object, **_kwargs: object):
        temp_seed_path = tmp_path / "temp_cluster_seeds.arrow"
        write_cluster_seeds_arrow(temp_seed_path, {"seed": "c_seed"})
        try:
            yield {"cluster_seeds": str(temp_seed_path)}
        finally:
            closed.append(True)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = False
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **kwargs: object):
            kwargs["prediction_state"].telemetry["incremental_seed_setup"] = {
                "seed_setup_cluster_seeds_source": "python"
            }
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

    def fail_raw_arrow_linker(*_args: object, **_kwargs: object):
        raise RuntimeError("raw Arrow linker failed")

    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fail_raw_arrow_linker,
    )
    monkeypatch.setattr(
        production_module,
        "temporary_cluster_seed_sidecars",
        fake_temporary_cluster_seed_sidecars,
    )
    _patch_fake_raw_arrow_planner(monkeypatch)

    with pytest.raises(RuntimeError, match="raw Arrow linker failed"):
        production_module.predict_incremental_promoted_linker_from_arrow(
            FakeClusterer(),
            ["seed", "query"],
            _direct_arrow_dataset(),
            arrow_dataset=_minimal_arrow_dataset(tmp_path),
            artifact=FakeArtifact(),
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
        )

    assert closed == [True]


@pytest.mark.parametrize("batching_threshold", [1, 2])
def test_query_disallow_resolution_is_batching_threshold_invariant_and_replans_complete_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batching_threshold: int,
) -> None:
    query_ids = ["q-require", "q-score"]
    captured: dict[str, Any] = {"rescored": []}

    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = FeaturizationInfo(features_to_use=["name_counts"])
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            seeds = {"seed-z": "c_seed_0", "seed-a": "c_seed_0"}
            original_inverse = {"c_seed": ["seed-z", "seed-a"]}
            split_inverse = {"c_seed_0": ["seed-z", "seed-a"]}
            return seeds, {"c_seed_0": "c_seed"}, original_inverse, split_inverse

        def _finish_incremental_with_seed_links(self, *args: object, **_kwargs: object):
            captured["linked"] = dict(cast(Mapping[str, str], args[2]))
            clusters = {"c_seed_0": ["seed-z", "seed-a"]}
            for signature_id, component_id in captured["linked"].items():
                clusters.setdefault(component_id, []).append(signature_id)
            return clusters

    class FakeFeaturizer:
        def __init__(self, signature_ids: tuple[str, ...]) -> None:
            self._signature_ids = signature_ids

        def signature_ids(self) -> list[str]:
            return list(self._signature_ids)

    arrow_dataset = _minimal_arrow_dataset(tmp_path)

    def fake_featurizer(dataset: ArrowDataset, **kwargs: object) -> FakeFeaturizer:
        captured.setdefault("featurizer_datasets", []).append(dataset)
        return FakeFeaturizer(tuple(cast(Any, kwargs["signature_ids"])))

    def fake_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        signature_ids = [str(value) for value in cast(list[str], kwargs["query_signature_ids"])]
        featurizer = cast(FakeFeaturizer, kwargs["rust_featurizer"])
        excluded = kwargs["cluster_seed_disallow_excluded_components"]
        decisions = []
        require_counts: list[float] = []
        if excluded is not None:
            assert signature_ids == ["q-score"]
            signature_id = signature_ids[0]
            captured["rescored"].append(signature_id)
            decisions.append(
                LinkOrAbstainDecision(
                    query_signature_index=featurizer.signature_ids().index(signature_id),
                    action="link",
                    row_index=0,
                    component_key="c_outside_retained_top_k",
                    score=0.60,
                    runner_up_score=None,
                    score_margin=None,
                )
            )
            require_counts.append(0.0)
        else:
            for row_index, signature_id in enumerate(signature_ids):
                decisions.append(
                    LinkOrAbstainDecision(
                        query_signature_index=featurizer.signature_ids().index(signature_id),
                        action="link",
                        row_index=row_index,
                        component_key="c_seed_0",
                        score=0.70 if signature_id == "q-require" else 0.99,
                        runner_up_score=None,
                        score_margin=None,
                    )
                )
                require_counts.append(1.0 if signature_id == "q-require" else 0.0)
        return SimpleNamespace(
            compact_result=SimpleNamespace(decisions=tuple(decisions)),
            pairwise_model_result=SimpleNamespace(row_signals={}),
            decision_row_signals={"constraint_require_count": np.asarray(require_counts, dtype=np.float32)},
            linked_signature_clusters={},
            telemetry={
                "query_count": len(signature_ids),
                "candidate_row_count": len(signature_ids),
                "pair_count": len(signature_ids),
            },
        )

    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(
            query_batch_size=min(int(kwargs["query_count"]), batching_threshold),
        ),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_dataset",
        fake_featurizer,
    )

    result = production_module.predict_incremental_promoted_linker_from_arrow(
        FakeClusterer(),
        ["seed-z", "seed-a", *query_ids],
        _direct_arrow_dataset(cluster_seeds_disallow={("q-require", "q-score")}),
        arrow_dataset=arrow_dataset,
        artifact=FakeArtifact(),
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=batching_threshold,
    )

    assert captured["linked"] == {
        "q-require": "c_seed_0",
        "q-score": "c_outside_retained_top_k",
    }
    assert captured["rescored"] == ["q-score"]
    assert captured["planner_disallows"] == [set()]
    assert captured["planner_plan_disallows"][-1] == {("q-score", "seed-a")}
    assert captured["featurizer_datasets"]
    assert all(dataset is arrow_dataset for dataset in captured["featurizer_datasets"])
    assert len(captured["featurizer_datasets"]) == len(captured["planner_plans"]) - 1
    assert result["clusters"] == {
        "c_seed_0": ["seed-z", "seed-a", "q-require"],
        "c_outside_retained_top_k": ["q-score"],
    }
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["global_query_disallow_rescore_count"] == 1
    assert telemetry["raw_arrow_batch_featurizer_count"] == len(captured["featurizer_datasets"])


def test_query_disallow_rescores_reuse_two_bounded_batch_featurizers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    query_ids = ["q0", "q1", "q2", "q3"]
    captured: dict[str, Any] = {
        "featurizer_builds": [],
        "featurizer_refs": [],
        "rescore_callback_refs": [],
        "rescore_context_refs": [],
        "rescored": [],
    }

    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            seeds = {"s0": "c0", "s1": "c1"}
            inverse = {"c0": ["s0"], "c1": ["s1"]}
            return seeds, {}, inverse, inverse

        def _finish_incremental_with_seed_links(self, *args: object, **_kwargs: object):
            captured["linked"] = dict(cast(Mapping[str, str], args[2]))
            captured["phase_a_released_before_finish"] = {
                "featurizers": all(reference() is None for reference in captured["featurizer_refs"]),
                "planner": all(reference() is None for reference in captured["planner_refs"]),
                "rescore_callback": all(reference() is None for reference in captured["rescore_callback_refs"]),
                "rescore_context": all(reference() is None for reference in captured["rescore_context_refs"]),
            }
            return {"done": [*query_ids]}

    class FakeFeaturizer:
        def __init__(self, signature_ids: tuple[str, ...]) -> None:
            self._signature_ids = signature_ids

        def signature_ids(self) -> list[str]:
            return list(self._signature_ids)

    def fake_featurizer(*_args: object, **kwargs: object) -> FakeFeaturizer:
        signature_ids = tuple(cast(Any, kwargs["signature_ids"]))
        captured["featurizer_builds"].append(signature_ids)
        featurizer = FakeFeaturizer(signature_ids)
        captured["featurizer_refs"].append(weakref.ref(featurizer))
        return featurizer

    initial_components = {"q0": "c0", "q1": "c0", "q2": "c1", "q3": "c1"}
    initial_scores = {"q0": 0.99, "q1": 0.80, "q2": 0.70, "q3": 0.60}
    rescored_components = {"q1": "c2", "q3": "c3"}

    def fake_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        batch = [str(value) for value in cast(Any, kwargs["query_signature_ids"])]
        featurizer = cast(FakeFeaturizer, kwargs["rust_featurizer"])
        is_rescore = kwargs["cluster_seed_disallow_excluded_components"] is not None
        decisions = []
        for signature_id in batch:
            if is_rescore:
                captured["rescored"].append(signature_id)
                component_key = rescored_components[signature_id]
                score = 0.50
            else:
                component_key = initial_components[signature_id]
                score = initial_scores[signature_id]
            decisions.append(
                LinkOrAbstainDecision(
                    query_signature_index=featurizer.signature_ids().index(signature_id),
                    action="link",
                    row_index=0,
                    component_key=component_key,
                    score=score,
                    runner_up_score=None,
                    score_margin=None,
                )
            )
        return SimpleNamespace(
            compact_result=SimpleNamespace(decisions=tuple(decisions)),
            pairwise_model_result=SimpleNamespace(row_signals={}),
            decision_row_signals={"constraint_require_count": np.zeros(len(batch), dtype=np.float32)},
            linked_signature_clusters={},
            telemetry={"query_count": len(batch), "candidate_row_count": 0, "pair_count": 0},
        )

    def fake_limits(**kwargs: object):
        return _mock_promoted_limits(
            query_batch_size=1,
            predicted_peak_delta_bytes=8_500,
        )

    real_rescore_context = production_module._QueryDisallowRescoreContext  # noqa: SLF001
    real_resolve_query_disallows = production_module._resolve_query_disallows_globally  # noqa: SLF001

    def capturing_rescore_context(**kwargs: Any) -> Any:
        context = real_rescore_context(**kwargs)
        captured["rescore_context_refs"].append(weakref.ref(context))
        return context

    def capturing_resolve_query_disallows(
        initial_decisions: Mapping[str, Any],
        disallow_partners: Mapping[str, set[str]],
        *,
        rescore: Any,
    ) -> tuple[dict[str, str], dict[str, int]]:
        captured["rescore_callback_refs"].append(weakref.ref(rescore))
        return real_resolve_query_disallows(initial_decisions, disallow_partners, rescore=rescore)

    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)
    monkeypatch.setattr(production_module, "_QueryDisallowRescoreContext", capturing_rescore_context)
    monkeypatch.setattr(production_module, "_resolve_query_disallows_globally", capturing_resolve_query_disallows)
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(production_module.feature_port, "build_rust_featurizer_from_arrow_dataset", fake_featurizer)

    result = production_module.predict_incremental_promoted_linker_from_arrow(
        FakeClusterer(),
        ["s0", "s1", *query_ids],
        _direct_arrow_dataset(
            cluster_seeds_disallow={("q0", "q1"), ("q2", "q3")},
        ),
        arrow_dataset=_minimal_arrow_dataset(tmp_path),
        artifact=FakeArtifact(),
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=1,
    )

    assert captured["planner_plans"][:4] == [("q0",), ("q1",), ("q2",), ("q3",)]
    assert captured["planner_plans"][4:] == [("q1",), ("q3",)]
    assert captured["rescored"] == ["q1", "q3"]
    assert len(captured["featurizer_builds"]) == 5
    assert captured["linked"] == {"q0": "c0", "q1": "c2", "q2": "c1", "q3": "c3"}
    assert captured["phase_a_released_before_finish"] == {
        "featurizers": True,
        "planner": True,
        "rescore_callback": True,
        "rescore_context": True,
    }
    assert result["incremental_linker_telemetry"]["raw_arrow_batch_featurizer_count"] == 5


def test_promoted_linker_reuses_one_plan_and_featurizer_for_four_scoring_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    query_ids = [f"q{index:02d}" for index in range(20)]
    captured: dict[str, Any] = {"featurizer_windows": [], "scored_batches": []}

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, *_args: object, **_kwargs: object):
            return {"c_seed": ["seed", *query_ids]}

    def fake_limits(**kwargs: object):
        query_count = int(kwargs["query_count"])
        max_query_batch_size = int(cast(int, kwargs["max_query_batch_size"]))
        return _mock_promoted_limits(
            query_batch_size=min(query_count, max_query_batch_size),
        )

    def fake_featurizer(*_args: object, **kwargs: object) -> object:
        captured["featurizer_windows"].append(tuple(cast(Any, kwargs["signature_ids"])))
        return object()

    def fake_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        batch = tuple(cast(Any, kwargs["query_signature_ids"]))
        captured["scored_batches"].append(batch)
        return SimpleNamespace(
            linked_signature_clusters={signature_id: "c_seed" for signature_id in batch},
            telemetry={"query_count": len(batch), "candidate_row_count": 0, "pair_count": 0},
        )

    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(production_module.feature_port, "build_rust_featurizer_from_arrow_dataset", fake_featurizer)

    result = production_module.predict_incremental_promoted_linker_from_arrow(
        FakeClusterer(),
        ["seed", *query_ids],
        _direct_arrow_dataset(),
        arrow_dataset=_minimal_arrow_dataset(tmp_path),
        artifact=FakeArtifact(),
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=2,
    )

    expected_windows = [tuple(query_ids[:8]), tuple(query_ids[8:16]), tuple(query_ids[16:])]
    assert captured["planner_plans"] == expected_windows
    assert captured["featurizer_windows"] == expected_windows
    assert captured["scored_batches"] == [tuple(query_ids[start : start + 2]) for start in range(0, 20, 2)]
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["raw_arrow_batch_plan_count"] == 3
    assert telemetry["raw_arrow_batch_featurizer_count"] == 3
    assert "raw_arrow_batch_plan_window_size" not in telemetry
    assert "raw_arrow_batch_featurizer_reused_batch_count" not in telemetry


def test_promoted_linker_replans_batch_when_post_featurizer_ram_limit_shrinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        artifact_dir = tmp_path
        retrieval_top_k = 25
        feature_columns = _PROMOTED_TEST_FEATURE_COLUMNS

    query_ids = [f"q{index:02d}" for index in range(20)]
    captured: dict[str, Any] = {
        "featurizer_batches": [],
        "limit_checks": [],
        "scored_batches": [],
    }

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, *_args: object, **_kwargs: object):
            return {"c_seed": ["seed", *query_ids]}

    def fake_limits(**kwargs: object):
        query_count = int(kwargs["query_count"])
        captured["limit_checks"].append(
            (
                bool(kwargs.get("retrieval_payload_resident", False)),
                len(captured.get("planner_plans", [])),
                len(captured["featurizer_batches"]),
            )
        )
        safe_size = 2 if captured["featurizer_batches"] else 10
        return _mock_promoted_limits(
            query_batch_size=min(query_count, safe_size),
        )

    def fake_featurizer(*_args: object, **kwargs: object) -> object:
        signature_ids = tuple(cast(Any, kwargs["signature_ids"]))
        captured["featurizer_batches"].append(signature_ids)
        return object()

    def fake_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        batch = tuple(cast(Any, kwargs["query_signature_ids"]))
        captured["scored_batches"].append(batch)
        return SimpleNamespace(
            linked_signature_clusters={signature_id: "c_seed" for signature_id in batch},
            telemetry={"query_count": len(batch), "candidate_row_count": len(batch), "pair_count": len(batch)},
        )

    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(production_module.feature_port, "build_rust_featurizer_from_arrow_dataset", fake_featurizer)

    result = production_module.predict_incremental_promoted_linker_from_arrow(
        FakeClusterer(),
        ["seed", *query_ids],
        _direct_arrow_dataset(),
        arrow_dataset=_minimal_arrow_dataset(tmp_path),
        artifact=FakeArtifact(),
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=10,
    )

    scored_signature_ids = [signature_id for batch in captured["scored_batches"] for signature_id in batch]
    assert scored_signature_ids == query_ids
    assert all(1 <= len(batch) <= 2 for batch in captured["scored_batches"])
    assert captured["limit_checks"][:5] == [
        (False, 0, 0),
        (False, 0, 0),
        (False, 0, 0),
        (True, 1, 0),
        (True, 1, 1),
    ]
    assert captured["featurizer_batches"] == captured["planner_plans"]
    assert captured["featurizer_batches"][0] == tuple(query_ids[:10])
    assert len(captured["featurizer_batches"]) < len(captured["scored_batches"])
    assert result["clusters"] == {"c_seed": ["seed", *query_ids]}
    assert result["incremental_linker_telemetry"]["raw_arrow_batch_memory_replan_count"] >= 1


def test_predict_from_arrow_uses_bound_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=object(),
        n_jobs=1,
    )
    valid_root = tmp_path / "valid"
    arrow_paths = write_minimal_arrow_prediction_bundle(valid_root)
    name_counts_index, _metrics = write_name_counts_index(valid_root, tiny_name_counts_tuple())
    clusterer.feature_contract["name_counts_manifest_sha256"] = hashlib.sha256(
        (Path(name_counts_index) / "manifest.json").read_bytes()
    ).hexdigest()
    arrow_paths["name_counts_index"] = name_counts_index
    write_test_arrow_artifact_manifest(valid_root, arrow_paths)
    arrow_dataset = ArrowDataset.open(valid_root, require_name_counts_index=True)
    captured: dict[str, Any] = {}

    def fake_build_rust_featurizer_from_arrow_dataset(dataset: ArrowDataset, **kwargs: Any) -> object:
        captured["dataset"] = dataset
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_dataset",
        fake_build_rust_featurizer_from_arrow_dataset,
    )
    monkeypatch.setattr(
        Clusterer,
        "_predict_from_rust_featurizer",
        lambda *_args, **_kwargs: ({"block": ["s1"]}, None),
    )

    clusterer.predict_from_arrow(
        {"block": ["s1"]},
        arrow_dataset,
    )

    assert captured["dataset"] is arrow_dataset
    assert arrow_dataset.native_name_counts_index is not None


def test_promoted_incremental_batch_telemetry_keeps_absolute_and_seed_fields_constant() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 40,
                "memory_stage_budget_bytes": 20,
                "raw_arrow_plan_seed_signature_count": 10,
                "raw_arrow_plan_cluster_count": 2,
            },
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 35,
                "memory_stage_budget_bytes": 20,
                "raw_arrow_plan_seed_signature_count": 10,
                "raw_arrow_plan_cluster_count": 2,
            },
        ],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 2
    assert merged["memory_total_ram_bytes"] == 100
    assert merged["memory_available_bytes"] == 40
    assert merged["memory_stage_budget_bytes"] == 20
    assert merged["memory_available_bytes_batch_conflict_count"] == 1
    assert merged["raw_arrow_plan_seed_signature_count"] == 10
    assert merged["raw_arrow_plan_cluster_count"] == 2


def test_promoted_incremental_batch_telemetry_records_constant_and_unknown_conflicts() -> None:
    cases = (
        (
            [
                {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10},
                {"query_count": 1, "raw_arrow_plan_seed_signature_count": 11},
            ],
            "raw_arrow_plan_seed_signature_count",
            10,
            1,
        ),
        (
            [
                {"query_count": 1, "custom_metric": "unregistered"},
                {"query_count": 1, "custom_metric": 3},
                {"query_count": 1, "custom_metric": 4},
            ],
            "custom_metric",
            "unregistered",
            2,
        ),
        (
            [{"query_count": 1, "new_metric": 3}, {"query_count": 1, "new_metric": 4}],
            "new_metric",
            3,
            1,
        ),
    )
    for batches, field, expected_value, expected_conflicts in cases:
        merged = production_module.merge_promoted_incremental_batch_telemetry(
            batches,
            batch_sizes=[1] * len(batches),
            configured_batch_size=1,
        )

        assert merged["query_count"] == len(batches)
        assert merged[field] == expected_value
        assert merged[f"{field}_batch_conflict_count"] == expected_conflicts


def _mock_promoted_limits(
    *,
    query_batch_size: int = 1,
    predicted_peak_delta_bytes: int = 2_000,
    predicted_peak_rss_bytes: int = 3_000,
) -> model_module.memory_budget.PromotedPhaseALimits:
    return model_module.memory_budget.PromotedPhaseALimits(
        query_batch_size=int(query_batch_size),
        predicted_peak_delta_bytes=int(predicted_peak_delta_bytes),
        predicted_peak_rss_bytes=int(predicted_peak_rss_bytes),
    )


def _build_minimal_incremental_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=object(),
        n_jobs=1,
    )


def _strict_rust_context(run_id: str = "test-strict-rust") -> SimpleNamespace:
    return SimpleNamespace(
        operation="cluster_predict",
        backend="rust",
        run_id=run_id,
    )


def test_subblocked_single_letter_groups_use_request_owned_seeds(monkeypatch) -> None:
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="transient_arrow_only",
            cluster_seeds_require={"seed": "seed_cluster"},
            cluster_seeds_disallow=set(),
            name_tuples=None,
        ),
    )

    def fake_predict_incremental(self, block_signatures, *_args, **_kwargs):
        del self
        assert dataset.cluster_seeds_require == {"seed": "seed_cluster"}
        assert _kwargs["prediction_state"].cluster_seeds_require is not dataset.cluster_seeds_require
        return {"clusters": {"single": list(block_signatures)}}

    monkeypatch.setattr(Clusterer, "_predict_incremental_python", fake_predict_incremental)
    result = clusterer._predict_subblocked_single_letter_incremental_groups(
        {"block|single": ["s1", "s2"]},
        pred_clusters={},
        dataset=dataset,
        partial_supervision={},
        runtime_context=cast(Any, _strict_rust_context("cleanup-no-dataset-featurizer")),
        total_ram_bytes=None,
    )

    assert result == {"single": ["s1", "s2"]}
    assert dataset.cluster_seeds_require == {"seed": "seed_cluster"}


def test_arrow_subblocked_single_letter_groups_reuse_and_extend_one_seed_map(tmp_path: Path) -> None:
    clusterer = _build_minimal_incremental_clusterer()
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    seed_map_ids: list[int] = []
    seed_map_snapshots: list[dict[str, str]] = []

    def fake_predict_incremental_from_arrow(
        block_signatures: list[str],
        dataset: object,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert dataset is arrow_dataset
        seed_map = cast(dict[str, str], kwargs["cluster_seeds_require"])
        seed_map_ids.append(id(seed_map))
        seed_map_snapshots.append(dict(seed_map))
        if block_signatures == ["q1"]:
            return {"clusters": {"seed_cluster": ["seed0", "seed1", "q1"]}}
        assert block_signatures == ["q2"]
        return {
            "clusters": {
                "seed_cluster": ["seed0", "seed1", "q1"],
                "new_cluster": ["q2"],
            }
        }

    clusterer.predict_incremental_from_arrow = cast(Any, fake_predict_incremental_from_arrow)

    result = clusterer._predict_subblocked_single_letter_groups_from_arrow(
        {"group_b": ["q2"], "group_a": ["q1"]},
        pred_clusters={"seed_cluster": ["seed0", "seed1"]},
        arrow_dataset=arrow_dataset,
        batching_threshold=10,
        partial_supervision={},
        runtime_context=cast(Any, _strict_rust_context("arrow-sequential-seed-map")),
        total_ram_bytes=None,
        name_tuples=frozenset(),
        cluster_seeds_disallow=set(),
        cluster_seeds_require={},
    )

    assert result == {
        "seed_cluster": ["seed0", "seed1", "q1"],
        "new_cluster": ["q2"],
    }
    assert seed_map_ids[0] == seed_map_ids[1]
    assert seed_map_snapshots == [
        {"seed0": "seed_cluster", "seed1": "seed_cluster"},
        {"seed0": "seed_cluster", "seed1": "seed_cluster", "q1": "seed_cluster"},
    ]


def test_next_unused_cluster_id_prevents_overwrite():
    pred_clusters = {
        "0": ["s0"],
        "1": ["s1"],
        "2": ["existing_singleton_cluster"],
    }
    start = model_module._next_unused_cluster_id(pred_clusters, 2)
    assert start == 3

    # Simulate the singleton recluster append loop in Python incremental prediction.
    for signatures in (["new_a"], ["new_b"]):
        cluster_id = model_module._next_unused_cluster_id(pred_clusters, start)
        pred_clusters[str(cluster_id)] = signatures
        start = cluster_id + 1

    assert pred_clusters["2"] == ["existing_singleton_cluster"]
    assert pred_clusters["3"] == ["new_a"]
    assert pred_clusters["4"] == ["new_b"]


def test_predict_incremental_without_seeds_covers_all_signatures(clusterer_dataset_factory):
    clusterer, dataset = clusterer_dataset_factory()
    dataset.cluster_seeds_require = {}
    block = ["3", "4", "5", "6", "7", "8"]

    output_no_subblock = _clusters(clusterer.predict_incremental(block, dataset))
    assigned_no_subblock = {signature for signatures in output_no_subblock.values() for signature in signatures}
    assert assigned_no_subblock == set(block)


def test_predict_incremental_batch_constraint_path_parity(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]

    baseline_clusterer, baseline_dataset = clusterer_dataset_factory()
    baseline = _clusters(baseline_clusterer.predict_incremental(block, baseline_dataset))

    batch_clusterer, batch_dataset = clusterer_dataset_factory()

    sig_ids = list(batch_dataset.signatures.keys())

    class _FakeIndexedFeaturizer:
        def signature_ids(self):
            return sig_ids

        def get_constraints_matrix_indexed(self, *_args, **_kwargs):
            return [None]

    calls = {"batch": 0}
    monkeypatch.setattr(
        model_module,
        "_initialize_incremental_constraint_backend",
        lambda *_args, **_kwargs: (_FakeIndexedFeaturizer(), True),
    )

    def _fake_get_constraints_matrix_indexed_rust(indexed_pairs, **kwargs):
        calls["batch"] += 1
        dont_merge = kwargs.get("dont_merge_cluster_seeds", True)
        incremental_flag = kwargs.get("incremental_dont_use_cluster_seeds", False)
        return [
            batch_dataset.get_constraint(
                sig_ids[i1],
                sig_ids[i2],
                dont_merge_cluster_seeds=dont_merge,
                incremental_dont_use_cluster_seeds=incremental_flag,
            )
            for i1, i2 in indexed_pairs
        ]

    monkeypatch.setattr(model_module, "get_constraints_matrix_indexed_rust", _fake_get_constraints_matrix_indexed_rust)

    batch_output = _clusters(batch_clusterer.predict_incremental(block, batch_dataset))
    assert _same_partition(batch_output, baseline), (
        f"Batch-constraint and baseline partitions differ:\n  batch={batch_output}\n  baseline={baseline}"
    )
    assert calls["batch"] > 0


def test_predict_subblocked_processes_subblocks_in_sorted_key_order(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory()
    block_signatures = ["3", "4", "5", "6"]
    observed_order: list[str] = []

    def _fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None, **kwargs):
        del signatures, anddata, maximum_size, first_k_letter_counts_sorted, kwargs
        # Intentionally unsorted insertion order to verify deterministic processing order in predict().
        return {"zeta": ["3", "4"], "alpha": ["5", "6"]}

    def _fake_predict_helper(
        self,
        block_dict,
        dataset,
        dists=None,
        cluster_model_params=None,
        partial_supervision=None,
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        runtime_context=None,
        total_ram_bytes=None,
        prediction_state=None,
    ):
        del self, dataset, dists, cluster_model_params, partial_supervision
        del use_s2_clusters, incremental_dont_use_cluster_seeds, runtime_context, total_ram_bytes
        key = next(iter(block_dict))
        observed_order.append(key)
        return {f"cluster_{len(observed_order)}": list(block_dict[key])}, None

    monkeypatch.setattr(model_module, "make_subblocks", _fake_make_subblocks)
    monkeypatch.setattr(model_module, "_signature_first_for_rules", lambda _: "john")
    monkeypatch.setattr(Clusterer, "predict_helper", _fake_predict_helper)

    clusterer.predict({"block": block_signatures}, dataset, batching_threshold=3)
    assert observed_order == ["block|subblock=alpha", "block|subblock=zeta"]


def test_clusterer_uses_graph_subblocking_directly_and_propagates_failures(
    clusterer_dataset_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_io_strict")

    class FailingGraph:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise FileNotFoundError("missing graph sidecar")

    graph = FailingGraph()
    monkeypatch.setattr(
        model_module,
        "make_dataset_graph_subblocking_cluster_fn",
        lambda *, config: graph,
    )

    selected = clusterer._subblocking_graph_cluster_fn()

    assert selected is graph
    with pytest.raises(FileNotFoundError, match="missing graph sidecar"):
        selected(["3", "4", "5"], dataset, target_subblock_size=2)


def test_best_incremental_cluster_respects_seed_score_mode():
    clusterer = _build_minimal_incremental_clusterer()
    cluster_dists = {
        "mean_favorite": (0.20, 2, 0.20),
        "min_favorite": (0.29, 2, 0.01),
    }

    clusterer.incremental_seed_score_mode = "mean"
    best_mean, best_mean_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_mean == "mean_favorite"
    assert best_mean_score == pytest.approx(0.20)

    clusterer.incremental_seed_score_mode = "min"
    best_min, best_min_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_min == "min_favorite"
    assert best_min_score == pytest.approx(0.01)

    clusterer.incremental_seed_score_mode = "mean_min_hybrid"
    clusterer.incremental_mean_min_hybrid_weight = 0.25
    best_hybrid_low, best_hybrid_low_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_hybrid_low == "mean_favorite"
    assert best_hybrid_low_score == pytest.approx(0.20)

    clusterer.incremental_mean_min_hybrid_weight = 0.75
    best_hybrid_high, best_hybrid_high_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_hybrid_high == "min_favorite"
    assert best_hybrid_high_score == pytest.approx(0.08)


def test_finish_incremental_with_seed_links_reclusters_only_abstains():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []
    residual_total_ram_bytes: list[int | None] = []

    def fake_predict_helper(
        block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None, prediction_state=None
    ):
        del dataset, partial_supervision, runtime_context
        residual_blocks.append(list(block_dict["block"]))
        residual_total_ram_bytes.append(total_ram_bytes)
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        type(
            "IncrementalDataset",
            (),
            {
                "cluster_seeds_require": {"seed0": "7", "seed1": "8"},
                "max_seed_cluster_id": 8,
                "signatures": {},
                "name_tuples": set(),
            },
        )(),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u1", "u2"],
        dataset,
        {"u1": "7_0"},
        {"7_0": "7"},
        {"7": ["seed0"], "8": ["seed1"]},
        False,
        {},
        runtime_context=cast(Any, object()),
        total_ram_bytes=123_456,
    )

    assert result == {"7": ["seed0", "u1"], "8": ["seed1"], "9": ["u2"]}
    assert residual_blocks == []
    assert residual_total_ram_bytes == []


def test_finish_incremental_with_seed_links_uses_seed_setup_when_dataset_seed_map_is_empty():
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={},
            signatures={},
            max_seed_cluster_id=0,
            name_tuples=None,
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["q1"],
        dataset,
        {"q1": "c1"},
        {},
        {"c1": ["s1", "s2"]},
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, object()),
    )

    assert result == {"c1": ["s1", "s2", "q1"]}


def test_finish_incremental_with_seed_links_reclusters_abstains_from_arrow(tmp_path: Path):
    clusterer = _build_minimal_incremental_clusterer()
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow residual Phase B should not call legacy predict_helper")

    def fake_predict_from_arrow(block_dict, dataset, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_dataset"] = dataset
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["cluster_seeds_disallow"] = set(kwargs["cluster_seeds_disallow"])
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fail_predict_helper)
    clusterer.predict_from_arrow = cast(Any, fake_predict_from_arrow)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "8"},
            cluster_seeds_disallow={("u2", "u3")},
            max_seed_cluster_id=8,
            signatures={},
            name_tuples=set(),
        ),
    )
    runtime_context = cast(Any, object())

    result = clusterer._finish_incremental_with_seed_links(
        ["u1", "u2", "u3"],
        dataset,
        {"u1": "7_0"},
        {"7_0": "7"},
        {"7": ["seed0"], "8": ["seed1"]},
        False,
        {},
        runtime_context=runtime_context,
        total_ram_bytes=123_456,
        arrow_dataset=arrow_dataset,
    )

    assert result == {"7": ["seed0", "u1"], "8": ["seed1"], "9": ["u2", "u3"]}
    assert captured["block_dict"] == {"block": ["u2", "u3"]}
    assert captured["arrow_dataset"] is arrow_dataset
    assert captured["partial_supervision"] == {("u2", "u3"): LARGE_DISTANCE}
    assert captured["cluster_seeds_disallow"] == {("u2", "u3")}
    assert captured["runtime_context"] is runtime_context
    assert captured["total_ram_bytes"] == 123_456


def test_finish_incremental_with_seed_links_splits_residual_phase_b_by_first_initial(tmp_path: Path):
    prediction_state = PredictionState()
    clusterer = _build_minimal_incremental_clusterer()
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    residual_blocks: list[list[str]] = []

    def fake_predict_from_arrow(block_dict, dataset, **kwargs):
        assert dataset is arrow_dataset
        del kwargs
        residual_block = list(block_dict["block"])
        residual_blocks.append(residual_block)
        return {"residual_cluster": residual_block}, None

    clusterer.predict_from_arrow = cast(Any, fake_predict_from_arrow)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed": "7"},
            cluster_seeds_disallow=set(),
            max_seed_cluster_id=7,
            signatures={
                "u_a1": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alice",
                    author_info_first="Alice",
                    author_info_orcid=None,
                ),
                "u_b1": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bob",
                    author_info_first="Bob",
                    author_info_orcid=None,
                ),
                "u_a2": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alan",
                    author_info_first="Alan",
                    author_info_orcid=None,
                ),
                "u_b2": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bea",
                    author_info_first="Bea",
                    author_info_orcid=None,
                ),
            },
            name_tuples=set(),
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u_a1", "u_b1", "u_a2", "u_b2"],
        dataset,
        {},
        {},
        {"7": ["seed"]},
        False,
        {},
        runtime_context=cast(Any, object()),
        arrow_dataset=arrow_dataset,
        prediction_state=prediction_state,
    )

    assert result == {"7": ["seed"], "8": ["u_a1", "u_a2"], "9": ["u_b1", "u_b2"]}
    assert residual_blocks == [["u_a1", "u_a2"], ["u_b1", "u_b2"]]
    assert prediction_state.telemetry["incremental_residual_phase_b"] == {
        "residual_phase_b_signature_count": 4,
        "residual_phase_b_group_count": 2,
        "residual_phase_b_pair_count_before": 6,
        "residual_phase_b_pair_count_after": 2,
        "residual_phase_b_pair_count_saved": 4,
    }


def test_finish_incremental_with_seed_links_residual_phase_b_preserves_same_orcid_group():
    prediction_state = PredictionState()
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []

    def fake_predict_helper(
        block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None, prediction_state=None
    ):
        del dataset, partial_supervision, runtime_context, total_ram_bytes
        residual_block = list(block_dict["block"])
        residual_blocks.append(residual_block)
        return {"residual_cluster": residual_block}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed": "7"},
            cluster_seeds_disallow=set(),
            max_seed_cluster_id=7,
            signatures={
                "u_a": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alice",
                    author_info_first="Alice",
                    author_info_orcid="0000-0000-0000-0001",
                ),
                "u_b": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bob",
                    author_info_first="Bob",
                    author_info_orcid="0000-0000-0000-0001",
                ),
            },
            name_tuples=set(),
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u_a", "u_b"],
        dataset,
        {},
        {},
        {"7": ["seed"]},
        False,
        {},
        runtime_context=cast(Any, object()),
        prediction_state=prediction_state,
    )

    assert result == {"7": ["seed"], "8": ["u_a", "u_b"]}
    assert residual_blocks == [["u_a", "u_b"]]
    assert prediction_state.telemetry["incremental_residual_phase_b"]["residual_phase_b_group_count"] == 1
    assert prediction_state.telemetry["incremental_residual_phase_b"]["residual_phase_b_pair_count_saved"] == 0


def test_build_incremental_seed_setup_uses_arrow_dataset_for_altered_profile_reclustering(tmp_path: Path):
    clusterer = _build_minimal_incremental_clusterer()
    arrow_dataset = _minimal_arrow_dataset(tmp_path)
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow altered-profile pre-splitting should not call legacy predict_helper")

    def fake_predict_from_arrow(block_dict, dataset, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_dataset"] = dataset
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["cluster_seeds_disallow"] = set(kwargs["cluster_seeds_disallow"])
        captured["incremental_dont_use_cluster_seeds"] = kwargs["incremental_dont_use_cluster_seeds"]
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {
            "altered_profile_0_0": ["seed0"],
            "altered_profile_0_1": ["seed1"],
            "altered_profile_1_0": ["seed2"],
            "altered_profile_1_1": ["seed3"],
        }, None

    clusterer.predict_helper = cast(Any, fail_predict_helper)
    clusterer.predict_from_arrow = cast(Any, fake_predict_from_arrow)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8", "seed3": "8", "seed4": "9"},
            cluster_seeds_disallow={("seed0", "seed1"), ("seed2", "seed3"), ("seed3", "seed4")},
            altered_cluster_signatures=["seed0", "seed2", "seed4"],
            name_tuples=None,
        ),
    )
    runtime_context = cast(Any, object())

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=runtime_context,
            total_ram_bytes=123_456,
            arrow_dataset=arrow_dataset,
        )
    )

    assert captured["block_dict"] == {
        "altered_profile_0": ["seed0", "seed1"],
        "altered_profile_1": ["seed2", "seed3"],
    }
    assert captured["arrow_dataset"] is arrow_dataset
    assert captured["partial_supervision"] == {
        ("seed0", "seed1"): LARGE_DISTANCE,
        ("seed2", "seed3"): LARGE_DISTANCE,
    }
    assert captured["cluster_seeds_disallow"] == {
        ("seed0", "seed1"),
        ("seed2", "seed3"),
        ("seed3", "seed4"),
    }
    assert captured["incremental_dont_use_cluster_seeds"] is True
    assert captured["runtime_context"] is runtime_context
    assert captured["total_ram_bytes"] == 123_456
    assert cluster_seeds_require == {
        "seed0": "7_0",
        "seed1": "7_1",
        "seed2": "8_0",
        "seed3": "8_1",
        "seed4": "9",
    }
    assert recluster_map == {"7_0": "7", "7_1": "7", "8_0": "8", "8_1": "8"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2", "seed3"], "9": ["seed4"]}
    assert split_inverse == {
        "7_0": ["seed0"],
        "7_1": ["seed1"],
        "8_0": ["seed2"],
        "8_1": ["seed3"],
        "9": ["seed4"],
    }
    assert split_inverse is not cluster_seeds_require_inverse


def test_altered_presplit_cache_name_tuple_key_computed_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    clusterer = _build_minimal_incremental_clusterer()
    arrow_dataset = _minimal_arrow_dataset(tmp_path)

    def fake_predict_from_arrow(block_dict, dataset, **kwargs):
        assert dataset is arrow_dataset
        del kwargs
        return {block_key: list(signature_ids) for block_key, signature_ids in block_dict.items()}, None

    clusterer.predict_from_arrow = cast(Any, fake_predict_from_arrow)
    name_tuples = frozenset({("anne", "ann")})
    calls = {"name_tuples_key": 0}
    real_cacheable_value = model_module._cacheable_value

    def counting_cacheable_value(value):
        if value is name_tuples:
            calls["name_tuples_key"] += 1
        return real_cacheable_value(value)

    monkeypatch.setattr(model_module, "_cacheable_value", counting_cacheable_value)

    def run(altered_cluster_signatures: list[str]) -> None:
        dataset = cast(
            ANDData,
            SimpleNamespace(
                cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8", "seed3": "8", "seed4": "9"},
                cluster_seeds_disallow=set(),
                altered_cluster_signatures=altered_cluster_signatures,
                name_tuples=name_tuples,
            ),
        )
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            total_ram_bytes=123_456,
            arrow_dataset=arrow_dataset,
        )

    # Two eligible multi-signature clusters ("7", "8") must share one computation.
    run(["seed0", "seed2", "seed4"])
    assert calls == {"name_tuples_key": 1}

    # A request whose only altered cluster is a singleton builds no cache key
    # and must not pay for the invariants at all.
    calls["name_tuples_key"] = 0
    run(["seed4"])
    assert calls == {"name_tuples_key": 0}


def test_build_incremental_seed_setup_normalizes_without_copying_source_seed_map() -> None:
    class ReadOnlySourceSeeds(dict[str, int]):
        def __deepcopy__(self, _memo: object) -> object:
            raise AssertionError("source seed map must not be deep-copied")

    source_seeds = ReadOnlySourceSeeds({"seed0": 7, "seed1": 7, "seed2": 8})
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require=source_seeds,
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=[],
            name_tuples=None,
        ),
    )
    clusterer = _build_minimal_incremental_clusterer()

    cluster_seeds_require, recluster_map, inverse, split_inverse = clusterer._build_incremental_seed_setup(
        dataset,
        {},
        runtime_context=cast(Any, object()),
    )

    assert cluster_seeds_require == {"seed0": "7", "seed1": "7", "seed2": "8"}
    assert inverse == {"7": ["seed0", "seed1"], "8": ["seed2"]}
    assert recluster_map == {}
    assert split_inverse == inverse
    assert split_inverse is inverse
    assert source_seeds == {"seed0": 7, "seed1": 7, "seed2": 8}


def test_build_incremental_seed_setup_reuses_owned_normalized_direct_arrow_map() -> None:
    source_seeds: dict[str, int | str] = {"seed0": "7", "seed1": "7", "seed2": "8"}
    dataset = production_module._DirectArrowIncrementalDataset(  # noqa: SLF001
        name_tuples=frozenset(),
        cluster_seeds_require=source_seeds,
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=[],
        max_seed_cluster_id=8,
        signatures={},
    )
    clusterer = _build_minimal_incremental_clusterer()

    cluster_seeds_require, recluster_map, inverse, split_inverse = clusterer._build_incremental_seed_setup(
        cast(ANDData, dataset),
        {},
        runtime_context=cast(Any, object()),
    )

    assert cluster_seeds_require is source_seeds
    assert cluster_seeds_require == {"seed0": "7", "seed1": "7", "seed2": "8"}
    assert recluster_map == {}
    assert split_inverse is inverse


def test_build_incremental_seed_setup_copies_direct_arrow_map_before_altered_processing() -> None:
    source_seeds: dict[str, int | str] = {"seed0": "7"}
    dataset = production_module._DirectArrowIncrementalDataset(  # noqa: SLF001
        name_tuples=frozenset(),
        cluster_seeds_require=source_seeds,
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=["seed0"],
        max_seed_cluster_id=7,
        signatures={},
    )
    clusterer = _build_minimal_incremental_clusterer()

    cluster_seeds_require, recluster_map, _inverse, _split_inverse = clusterer._build_incremental_seed_setup(
        cast(ANDData, dataset),
        {},
        runtime_context=cast(Any, object()),
    )

    assert cluster_seeds_require == source_seeds
    assert cluster_seeds_require is not source_seeds
    assert recluster_map == {}


def test_predict_from_rust_featurizer_does_not_posthoc_merge_when_incremental_dont_use_cluster_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=object(),
        cluster_model=object(),
        n_jobs=1,
        use_default_constraints_as_supervision=True,
    )
    captured: dict[str, Any] = {}

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return [("s1", "seeded"), ("s2", "seeded")]

        def signature_rule_metadata(self):
            return []

    def fake_cluster_one_block_with_logging(
        self,
        block_signatures,
        dist_matrix,
        cluster_model_params,
        dataset,
        all_disallow_signature_ids,
        *,
        block_key,
        incremental_dont_use_cluster_seeds,
        prediction_state=None,
    ):
        del self, block_signatures, dist_matrix, cluster_model_params, all_disallow_signature_ids, block_key
        captured["cluster_seeds_require"] = dict(dataset.cluster_seeds_require)
        return [0, 1]

    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block_with_logging)

    pred_clusters, _ = clusterer.predict_from_rust_featurizer(
        {"block": ["s1", "s2"]},
        FakeRustFeaturizer(),
        dists={"block": np.asarray([[0.0, 1.0], [1.0, 0.0]])},
        incremental_dont_use_cluster_seeds=True,
    )

    assert captured["cluster_seeds_require"] == {}
    assert _same_partition(pred_clusters, {"a": ["s1"], "b": ["s2"]})


def test_partial_supervision_disallow_merge_respects_reverse_existing_pair():
    dataset = SimpleNamespace(cluster_seeds_disallow={("q", "s1")})

    merged = model_module._partial_supervision_with_cluster_seed_disallows(
        ["q", "s1"],
        dataset,
        {("s1", "q"): 42.0},
        cluster_seed_disallows={("q", "s1")},
    )

    assert merged == {("s1", "q"): 42.0}


def test_build_incremental_seed_setup_rejects_altered_signature_missing_seed():
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["missing_seed"],
            name_tuples=None,
        ),
    )

    with pytest.raises(ValueError, match="must all be present in cluster_seeds_require"):
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
        )


def test_top1_consensus_broadcast_only_applies_when_cluster_members_agree():
    def _run(
        mode: Literal["always", "never", "top1_consensus"],
        signature_dists: dict[str, dict[int, tuple[float, int, float]]],
    ) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()
        clusterer.incremental_precluster_broadcast_mode = mode

        def fake_predict_helper(
            block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None, prediction_state=None
        ):
            del dataset, partial_supervision, runtime_context, total_ram_bytes
            if "incremental_unassigned" in block_dict:
                return {"incremental_cluster": list(block_dict["incremental_unassigned"])}, None
            if "block" in block_dict:
                return {"singleton_cluster": list(block_dict["block"])}, None
            raise AssertionError(f"Unexpected block_dict={block_dict}")

        clusterer.predict_helper = cast(Any, fake_predict_helper)
        dataset = cast(
            ANDData,
            type(
                "IncrementalDataset",
                (),
                {
                    "cluster_seeds_require": {"seed0": 0, "seed1": 1},
                    "max_seed_cluster_id": 2,
                    "signatures": {},
                    "name_tuples": set(),
                },
            )(),
        )
        signature_to_cluster_to_average_dist = cast(
            dict[str, dict[int | str, IncrementalDistStats]],
            {signature_id: cluster_dists.copy() for signature_id, cluster_dists in signature_dists.items()},
        )
        return clusterer._run_incremental_phases_bcd(
            ["u1", "u2"],
            dataset,
            signature_to_cluster_to_average_dist,
            dict(dataset.cluster_seeds_require),
            {},
            {0: ["seed0"], 1: ["seed1"]},
            False,
            {},
            runtime_context=cast(Any, object()),
        )

    divergent_top1_dists = {
        "u1": {0: (0.10, 1, 0.10), 1: (0.60, 1, 0.60)},
        "u2": {0: (0.60, 1, 0.60), 1: (0.20, 1, 0.20)},
    }
    always_divergent = _run("always", divergent_top1_dists)
    never_divergent = _run("never", divergent_top1_dists)
    consensus_divergent = _run("top1_consensus", divergent_top1_dists)
    assert always_divergent == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}
    assert never_divergent == {"0": ["seed0", "u1"], "1": ["seed1", "u2"]}
    assert consensus_divergent == never_divergent

    consensus_top1_dists = {
        "u1": {0: (0.10, 1, 0.10), 1: (0.60, 1, 0.60)},
        "u2": {0: (0.70, 1, 0.70), 1: (0.80, 1, 0.80)},
    }
    never_consensus = _run("never", consensus_top1_dists)
    consensus_enabled = _run("top1_consensus", consensus_top1_dists)
    assert never_consensus == {"0": ["seed0", "u1"], "1": ["seed1"], "2": ["u2"]}
    assert consensus_enabled == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}


def test_precluster_broadcast_preserves_min_score_semantics():
    def _run(
        *,
        seed_score_mode: Literal["min", "mean_min_hybrid"],
        mean_min_hybrid_weight: float = 0.5,
    ) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()
        clusterer.incremental_precluster_broadcast_mode = "always"
        clusterer.incremental_seed_score_mode = seed_score_mode
        clusterer.incremental_mean_min_hybrid_weight = mean_min_hybrid_weight

        def fake_predict_helper(
            block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None, prediction_state=None
        ):
            del dataset, partial_supervision, runtime_context, total_ram_bytes
            if "incremental_unassigned" in block_dict:
                return {"incremental_cluster": list(block_dict["incremental_unassigned"])}, None
            if "block" in block_dict:
                return {"singleton_cluster": list(block_dict["block"])}, None
            raise AssertionError(f"Unexpected block_dict={block_dict}")

        clusterer.predict_helper = cast(Any, fake_predict_helper)
        dataset = cast(
            ANDData,
            type(
                "IncrementalDataset",
                (),
                {
                    "cluster_seeds_require": {"seed0": 0, "seed1": 1},
                    "max_seed_cluster_id": 2,
                    "signatures": {},
                    "name_tuples": set(),
                },
            )(),
        )
        signature_to_cluster_to_average_dist = cast(
            dict[str, dict[int | str, IncrementalDistStats]],
            {
                "u1": {0: (0.40, 1, 0.01), 1: (0.20, 1, 0.20)},
                "u2": {0: (0.40, 1, 0.80), 1: (0.20, 1, 0.20)},
            },
        )
        return clusterer._run_incremental_phases_bcd(
            ["u1", "u2"],
            dataset,
            signature_to_cluster_to_average_dist,
            dict(dataset.cluster_seeds_require),
            {},
            {0: ["seed0"], 1: ["seed1"]},
            False,
            {},
            runtime_context=cast(Any, object()),
        )

    min_result = _run(seed_score_mode="min")
    assert min_result == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}

    hybrid_result = _run(seed_score_mode="mean_min_hybrid", mean_min_hybrid_weight=0.75)
    assert hybrid_result == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}
