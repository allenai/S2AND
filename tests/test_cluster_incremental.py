import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.incremental_linking.production as production_module
import s2and.model as model_module
from s2and.arrow_inputs import ValidatedArrowInputs, validate_arrow_prediction_artifacts
from s2and.consts import LARGE_DISTANCE, NORMALIZATION_VERSION
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import (
    read_incremental_query_signatures_arrow,
    write_altered_cluster_signatures_arrow,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_cluster_seed_disallows_arrow,
    write_cluster_seeds_arrow,
    write_name_counts_index,
)
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
    RawArrowPlanBundle,
)
from s2and.model import Clusterer, IncrementalDistStats
from tests.helpers import (
    tiny_name_counts_index,
    tiny_name_counts_provenance,
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


def _minimal_arrow_paths(tmp_path: Path, *, include_cluster_seeds: bool = False) -> dict[str, str]:
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    if include_cluster_seeds:
        cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
        write_cluster_seeds_arrow(cluster_seeds_path, {})
        paths["cluster_seeds"] = str(cluster_seeds_path)
    write_test_arrow_artifact_manifest(tmp_path, paths)
    return paths


def _validated_promoted_arrow_paths(paths: Mapping[str, Any]) -> ValidatedArrowInputs:
    return validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index="name_counts_index" in paths,
        expected_normalization_version=NORMALIZATION_VERSION,
    )


def _attach_minimal_arrow_paths(dataset: Any, tmp_path: Path) -> dict[str, str]:
    paths = _minimal_arrow_paths(tmp_path)
    dataset.arrow_paths = paths
    return paths


def test_raw_arrow_plan_windows_isolate_seed_overlap_queries() -> None:
    windows = production_module._raw_arrow_plan_windows(
        ["query-1", "seed-1", "query-2", "query-3", "seed-2", "query-4"],
        window_size=2,
        seed_signature_ids={"seed-1", "seed-2"},
    )

    assert windows == [["query-1"], ["seed-1"], ["query-2", "query-3"], ["seed-2"], ["query-4"]]


def _patch_fake_raw_arrow_planner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, Any] | None = None,
) -> None:
    """Install a planner-shaped Rust fake for promoted raw Arrow orchestration tests."""

    class FakePlanner:
        def __init__(self, _paths: object, query_signature_ids: list[str], **_kwargs: object):
            self._paths = cast(dict[str, str], _paths)
            self._query_signature_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_inits", []).append(self._query_signature_ids)

        @classmethod
        def from_query_signatures(cls, paths: object, **kwargs: object) -> "FakePlanner":
            path_map = cast(dict[str, str], paths)
            rows = read_incremental_query_signatures_arrow(Path(path_map["query_signatures"]))
            return cls(path_map, [row.signature_id for row in rows], **kwargs)

        @classmethod
        def from_auto_queries(cls, paths: object, **kwargs: object) -> "FakePlanner":
            return cls(paths, [], **kwargs)

        def build_telemetry(self):
            return {
                "query_signature_count": len(self._query_signature_ids),
                "signature_count": len(self._query_signature_ids),
            }

        def plan(self, query_signature_ids: list[str], **_kwargs: object):
            query_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_plans", []).append(query_ids)
            plan: dict[str, Any] = {
                "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
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
        "build_rust_featurizer_from_arrow_paths",
        lambda *_args, **_kwargs: object(),
    )


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


def test_finish_incremental_lazily_resolves_filtered_name_tuples_for_direct_arrow_dataset() -> None:
    """Direct Arrow's default name_tuples token should not crash compatibility checks."""

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
        name_tuples="filtered",
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


def test_sidecar_cache_fingerprint_hashes_full_content_at_equal_size_and_mtime(tmp_path: Path) -> None:
    path = tmp_path / "cluster_seeds.arrow"
    path.write_bytes(b"a" * (1024 * 1024))
    original_stat = path.stat()
    first = model_module._path_cache_fingerprint(path)

    with path.open("r+b") as output:
        output.seek(200_000)
        output.write(b"changed")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = model_module._path_cache_fingerprint(path)
    assert second[:3] == first[:3]
    assert second[3] != first[3]


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


def test_cluster_seed_inverse_canonicalizes_component_ids() -> None:
    assert model_module._cluster_seeds_require_inverse({"s1": 7, "s2": "7"}) == {"7": ["s1", "s2"]}


def test_predict_from_rust_featurizer_proxy_exposes_signature_rule_metadata() -> None:
    captured: dict[str, Any] = {}

    class DummyClusterer:
        predict_from_rust_featurizer = Clusterer.predict_from_rust_featurizer

        def predict_helper(self, block_dict, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            return {"block_0": list(block_dict["block"])}, kwargs["dists"]

    class FakeRustFeaturizer:
        def signature_rule_metadata(self):
            return [
                ("s_alice", "Alice", "0000-0000-0000-0001"),
                ("s_bob", "Bob", None),
                ("s_alicia", "Alicia", "0000-0000-0000-0001"),
            ]

    clusterer = DummyClusterer()
    cast(Any, clusterer).predict_from_rust_featurizer(
        {"block": ["s_alice", "s_bob", "s_alicia"]},
        FakeRustFeaturizer(),
        dists={"block": np.asarray([0.1, 0.2, 0.3], dtype=np.float64)},
        cluster_seeds_require={},
    )

    proxy_dataset = captured["dataset"]
    assert proxy_dataset.signatures["s_alice"].author_info_first == "Alice"
    groups = model_module._residual_phase_b_first_initial_groups(
        SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False),
        proxy_dataset,
        ["s_alice", "s_bob"],
        {},
    )
    assert groups == [["s_alice"], ["s_bob"]]
    assert model_module._can_skip_orcid_homogeneous_altered_presplit(
        SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False),
        proxy_dataset,
        ["s_alice", "s_alicia"],
        {},
        set(),
    )


def _seeds_preserved(clusters: dict[str, list[str]], seed_groups: list[list[str]]) -> bool:
    """Each seed group must be entirely contained in one predicted cluster."""
    cluster_sets = [set(sigs) for sigs in clusters.values() if sigs]
    for group in seed_groups:
        group_set = set(group)
        if not any(group_set.issubset(cluster_set) for cluster_set in cluster_sets):
            return False
    return True


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
        use_cache=False,
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


def test_predict_incremental_old_helper_name_is_not_exposed():
    legacy_helper_name = "_predict_incremental_" "helper"
    assert not hasattr(Clusterer, legacy_helper_name)


def test_promoted_incremental_orcid_fanout_by_query_counts_matching_components() -> None:
    dataset = SimpleNamespace(
        signatures={
            "q": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "blank": SimpleNamespace(author_info_orcid="   "),
            "other": SimpleNamespace(author_info_orcid="0000-0000-0000-0002"),
            "seed_a": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "seed_b": SimpleNamespace(author_info_orcid="0000-0000-0000-0001"),
            "seed_c": SimpleNamespace(author_info_orcid="0000-0000-0000-0003"),
            "seed_blank": SimpleNamespace(author_info_orcid="   "),
        }
    )
    fanout = production_module.promoted_incremental_orcid_fanout_by_query(
        dataset,  # type: ignore[arg-type]
        ["q", "blank", "other"],
        {"seed_a": "cluster_a", "seed_b": "cluster_b", "seed_c": "cluster_b", "seed_blank": "cluster_blank"},
        orcid_enabled=True,
    )

    assert fanout == {"q": (2, 3)}
    assert (
        production_module.promoted_incremental_orcid_fanout_by_query(
            dataset,  # type: ignore[arg-type]
            ["q"],
            {"seed_a": "cluster_a"},
            orcid_enabled=False,
        )
        == {}
    )


def test_promoted_incremental_orcid_fanout_skips_seed_scan_without_query_orcids(monkeypatch):
    calls: list[str] = []

    def fake_signature_orcid(_dataset, signature_id):
        calls.append(str(signature_id))
        if str(signature_id).startswith("seed"):
            raise AssertionError("seed scan should be skipped when no query has an ORCID")
        return None

    monkeypatch.setattr(production_module, "_signature_orcid", fake_signature_orcid)

    fanout = production_module.promoted_incremental_orcid_fanout_by_query(
        cast(ANDData, SimpleNamespace()),
        ["query-1", "query-2"],
        {"seed-1": "component-1"},
        orcid_enabled=True,
    )

    assert fanout == {}
    assert calls == ["query-1", "query-2"]


def test_predict_incremental_from_arrow_paths_uses_promoted_linker_without_dataset(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow")
    clusterer.incremental_linker_artifact_dir = tmp_path
    arrow_paths = _minimal_arrow_paths(tmp_path)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"3": "0", "4": "0"})
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    block = ["3", "4", "5", "6"]
    runtime_context = cast(Any, SimpleNamespace(backend="rust", run_id="test-direct-arrow"))
    captured: dict[str, Any] = {}
    payload = {
        "clusters": {"0": ["3", "4", "5"], "1": ["6"]},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    def fake_load_signature_info(paths: Mapping[str, Any], signature_ids: Any) -> dict[str, Any]:
        captured["metadata_paths"] = dict(paths)
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
        "predict_incremental_promoted_linker_from_arrow_paths",
        fake_promoted_linker,
    )

    result = clusterer.predict_incremental_from_arrow_paths(
        block,
        arrow_paths,
        batching_threshold=2,
        total_ram_bytes=100_000,
        runtime_context=runtime_context,
        name_tuples={("alex", "al")},
    )

    assert result == payload
    assert captured["clusterer"] is clusterer
    assert captured["block_signatures"] == block
    assert not isinstance(captured["dataset"], ANDData)
    assert captured["dataset"].cluster_seeds_require == {}
    assert captured["dataset"].name_tuples == {("al", "alex")}
    assert captured["dataset"].arrow_paths["cluster_seeds"] == str(cluster_seeds_path)
    assert captured["arrow_paths"]["cluster_seeds"] == str(cluster_seeds_path)
    assert isinstance(captured["arrow_paths"], ValidatedArrowInputs)
    assert captured["runtime_context"] is runtime_context
    assert captured["batching_threshold"] == 2
    assert captured["total_ram_bytes"] == 100_000
    assert captured["metadata_signature_ids"] == ("5", "6")
    assert set(captured["dataset"].signatures) == {"5", "6"}


def test_predict_incremental_from_arrow_paths_accepts_explicit_seed_mapping(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_seed_mapping")
    clusterer.incremental_linker_artifact_dir = tmp_path
    arrow_paths = _minimal_arrow_paths(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_signature_info(_paths: Mapping[str, Any], signature_ids: Any) -> dict[str, Any]:
        captured["metadata_signature_ids"] = tuple(signature_ids)
        return {}

    def fake_promoted_linker(_clusterer: Any, _block_signatures: list[str], dataset_arg: Any, **_kwargs: Any):
        captured["dataset"] = dataset_arg
        return {"clusters": {"0": ["3", "4", "5"]}}

    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow_paths",
        fake_promoted_linker,
    )

    result = clusterer.predict_incremental_from_arrow_paths(
        ["3", "4", "5"],
        arrow_paths,
        cluster_seeds_require={"3": 0, "4": 0},
    )

    assert result["clusters"] == {"0": ["3", "4", "5"]}
    assert captured["dataset"].cluster_seeds_require == {"3": "0", "4": "0"}
    assert captured["metadata_signature_ids"] == ("5",)


def test_predict_incremental_from_arrow_paths_requires_seed_source(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_no_seeds")
    arrow_paths = _minimal_arrow_paths(tmp_path)
    monkeypatch.setattr(
        model_module,
        "predict_incremental_promoted_linker_from_arrow_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("promoted linker should not run")),
    )

    with pytest.raises(model_module.MissingArrowArtifactError, match="cluster_seeds_source"):
        clusterer.predict_incremental_from_arrow_paths(["3", "4", "5"], arrow_paths)


def test_predict_incremental_from_arrow_paths_loads_altered_seed_metadata(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_altered")
    clusterer.incremental_linker_artifact_dir = tmp_path
    arrow_paths = _minimal_arrow_paths(tmp_path)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"3": "0", "4": "0", "8": "1"})
    write_altered_cluster_signatures_arrow(altered_path, ["3"])
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    arrow_paths["altered_cluster_signatures"] = str(altered_path)
    captured: dict[str, Any] = {}

    def fake_load_signature_info(_paths: Mapping[str, Any], signature_ids: Any) -> dict[str, Any]:
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
        "predict_incremental_promoted_linker_from_arrow_paths",
        fake_promoted_linker,
    )

    clusterer.predict_incremental_from_arrow_paths(["3", "4", "5"], arrow_paths)

    assert set(captured["metadata_signature_ids"]) == {"3", "4", "5"}
    assert captured["dataset"].altered_cluster_signatures is None
    assert set(captured["dataset"].signatures) == {"3", "4", "5"}


def test_predict_incremental_from_arrow_paths_loads_seed_orcids_for_budget_floor(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_incremental_direct_arrow_orcid_floor")
    clusterer.incremental_linker_artifact_dir = tmp_path
    arrow_paths = _minimal_arrow_paths(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_signature_info(_paths: Mapping[str, Any], signature_ids: Any) -> dict[str, Any]:
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

    def fake_load_orcid_info(_paths: Mapping[str, Any], signature_ids: Any) -> dict[str, Any]:
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
        "predict_incremental_promoted_linker_from_arrow_paths",
        fake_promoted_linker,
    )

    clusterer.predict_incremental_from_arrow_paths(
        ["3", "4", "5", "8"],
        arrow_paths,
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

    signatures_path = tmp_path / "signatures.arrow"
    schema = pa.schema(
        [
            ("signature_id", pa.string()),
            ("paper_id", pa.string()),
            ("author_first", pa.string()),
            ("author_middle", pa.string()),
            ("author_last", pa.string()),
            ("author_orcid", pa.string()),
        ]
    )
    batches = [
        pa.record_batch(
            [
                pa.array(["s1"], type=pa.string()),
                pa.array(["p1"], type=pa.string()),
                pa.array(["Alice"], type=pa.string()),
                pa.array([""], type=pa.string()),
                pa.array(["Jones"], type=pa.string()),
                pa.array([None], type=pa.string()),
            ],
            schema=schema,
        ),
        pa.record_batch(
            [
                pa.array(["s2", "s3"], type=pa.string()),
                pa.array(["p2", None], type=pa.string()),
                pa.array(["Anne-Marie", "Null"], type=pa.string()),
                pa.array(["", ""], type=pa.string()),
                pa.array(["Ng", "Paper"], type=pa.string()),
                pa.array(["0000-0002", None], type=pa.string()),
            ],
            schema=schema,
        ),
    ]
    with pa.OSFile(str(signatures_path), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
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

    signatures = model_module._load_arrow_incremental_signature_info(
        {"signatures": str(signatures_path), "signatures_batch_index": str(index_path)},
        ["s2", "s3"],
    )

    assert set(signatures) == {"s2", "s3"}
    assert signatures["s2"].paper_id == "p2"
    assert signatures["s2"].author_info_first == "Anne-Marie"
    assert signatures["s2"].author_info_last == "Ng"
    assert signatures["s2"].author_info_orcid == "0000-0002"
    assert signatures["s3"].paper_id is None
    assert read_batch_indices == [1]

    read_batch_indices.clear()
    orcid_signatures = model_module._load_arrow_incremental_orcid_signature_info(
        {"signatures": str(signatures_path), "signatures_batch_index": str(index_path)},
        ["s2"],
    )
    assert set(orcid_signatures) == {"s2"}
    assert orcid_signatures["s2"].paper_id is None
    assert orcid_signatures["s2"].author_info_first is None
    assert orcid_signatures["s2"].author_info_orcid == "0000-0002"
    assert read_batch_indices == [1]


def test_load_arrow_incremental_orcid_signature_info_skips_missing_seed_ids(tmp_path: Path) -> None:
    import pyarrow as pa

    signatures_path = tmp_path / "signatures.arrow"
    table = pa.table(
        {
            "signature_id": pa.array(["query"], type=pa.string()),
            "author_orcid": pa.array(["0000-0000-0000-0001"], type=pa.string()),
        }
    )
    with pa.OSFile(str(signatures_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(
        signatures_path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
    )

    assert (
        model_module._load_arrow_incremental_orcid_signature_info(
            {"signatures": str(signatures_path), "signatures_batch_index": str(index_path)},
            ["stale_seed"],
        )
        == {}
    )


def test_load_arrow_incremental_orcid_signature_info_accepts_missing_orcid_column(tmp_path: Path) -> None:
    import pyarrow as pa

    signatures_path = tmp_path / "signatures.arrow"
    write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["query"], type=pa.string())}),
        signatures_path,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(
        signatures_path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
    )

    assert (
        model_module._load_arrow_incremental_orcid_signature_info(
            {"signatures": str(signatures_path), "signatures_batch_index": str(index_path)},
            ["query"],
        )
        == {}
    )


def test_seed_cluster_count_matches_anddata_cluster_count() -> None:
    assert model_module._seed_cluster_count_from_seed_map({"s1": "7", "s2": "7", "s3": "9"}) == 2
    assert model_module._seed_cluster_count_from_seed_map({"s1": "component_a", "s2": "component_b"}) == 2


def test_cached_incremental_name_tuples_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    model_module._load_name_tuples_for_incremental_rules.cache_clear()
    loaded: list[str] = []

    def fake_load(filename: str) -> set[tuple[str, str]]:
        loaded.append(filename)
        return {("bill", "william")}

    monkeypatch.setattr(
        model_module,
        "_load_name_tuples_from_file",
        fake_load,
    )
    try:
        name_tuples = model_module._name_tuples_for_incremental_rules("filtered")
        default_name_tuples = model_module._name_tuples_for_incremental_rules(None)

        assert name_tuples == frozenset({("bill", "william")})
        assert default_name_tuples is name_tuples
        assert loaded == ["s2and_name_tuples_canonical.txt"]
        assert model_module._load_name_tuples_for_incremental_rules() is name_tuples
        assert not hasattr(name_tuples, "add")
    finally:
        model_module._load_name_tuples_for_incremental_rules.cache_clear()


def test_bare_clusterer_cannot_select_an_incremental_linker_from_another_bundle() -> None:
    with pytest.raises(FileNotFoundError, match="requires an attached incremental linker artifact"):
        model_module._required_incremental_linker_artifact_dir(SimpleNamespace())

    attached = SimpleNamespace(artifact_dir=Path("explicit-linker"))
    assert model_module._required_incremental_linker_artifact_dir(
        SimpleNamespace(incremental_linker_artifact=attached)
    ) == Path("explicit-linker")


def test_predict_incremental_arrow_promoted_linker_cleans_up_temp_seed_context_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []

    class FakeArtifact:
        metadata = SimpleNamespace(
            retrieval_top_k=25,
            feature_columns=_PROMOTED_TEST_FEATURE_COLUMNS,
        )

    @contextmanager
    def fake_temporary_arrow_paths_with_cluster_seeds(*_args: object, **_kwargs: object):
        temp_seed_path = tmp_path / "temp_cluster_seeds.arrow"
        write_cluster_seeds_arrow(temp_seed_path, {"seed": "c_seed"})
        try:
            yield {
                **_minimal_arrow_paths(tmp_path),
                "cluster_seeds": str(temp_seed_path),
            }
        finally:
            closed.append(True)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = False
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {"normalization_version": NORMALIZATION_VERSION}
        _last_incremental_seed_setup_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            self._last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "python"}
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

    def fail_raw_arrow_linker(*_args: object, **_kwargs: object):
        raise RuntimeError("raw Arrow linker failed")

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(production_module, "clusterer_uses_name_count_features", lambda _clusterer: False)
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fail_raw_arrow_linker,
    )
    monkeypatch.setattr(
        production_module,
        "temporary_arrow_paths_with_cluster_seeds",
        fake_temporary_arrow_paths_with_cluster_seeds,
    )
    _patch_fake_raw_arrow_planner(monkeypatch)

    with pytest.raises(RuntimeError, match="raw Arrow linker failed"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            FakeClusterer(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples=set())),
            arrow_paths=_validated_promoted_arrow_paths(_minimal_arrow_paths(tmp_path)),
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
        )

    assert closed == [True]


def test_predict_incremental_arrow_promoted_linker_uses_cached_artifact_and_typed_request_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(
            retrieval_top_k=25,
            feature_columns=_PROMOTED_TEST_FEATURE_COLUMNS,
        )
        artifact_dir = tmp_path

    captured: dict[str, Any] = {}
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    disallow_path = tmp_path / "cluster_seed_disallows.arrow"
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"seed": "c_seed"})
    write_cluster_seed_disallows_arrow(disallow_path, [("query", "seed")])
    write_altered_cluster_signatures_arrow(altered_path, ["seed"])
    arrow_paths = {
        **_minimal_arrow_paths(tmp_path),
        "cluster_seeds": str(cluster_seeds_path),
        "cluster_seed_disallows": str(disallow_path),
        "altered_cluster_signatures": str(altered_path),
    }

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {"normalization_version": NORMALIZATION_VERSION}
        _last_incremental_seed_setup_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, dataset: Any, *_args: object, **kwargs: object):
            seed_arrow_paths = cast(Mapping[str, Any], kwargs["arrow_paths"])
            captured["seed_setup_arrow_paths"] = dict(seed_arrow_paths)
            captured["altered_from_arrow"] = model_module._dataset_altered_cluster_signatures(
                dataset,
                seed_arrow_paths,
            )
            self._last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, *_args: object, **kwargs: object):
            captured["finish_arrow_paths"] = dict(cast(Mapping[str, Any], kwargs["arrow_paths"]))
            return {"c_seed": ["seed", "query"]}

    def fake_raw_arrow_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["raw_arrow_kwargs"] = dict(kwargs)
        return SimpleNamespace(
            linked_signature_clusters={"query": "c_seed"},
            telemetry={"query_count": 1, "candidate_row_count": 1, "pair_count": 1},
        )

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: pytest.fail("a supplied production artifact must not be loaded again"),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(query_count=int(kwargs["query_count"]), query_batch_size=1),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_raw_arrow_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)

    result = production_module.predict_incremental_promoted_linker_from_arrow_paths(
        FakeClusterer(),
        ["seed", "query"],
        cast(
            ANDData,
            SimpleNamespace(
                name_tuples=set(),
                cluster_seeds_disallow=set(),
                altered_cluster_signatures=None,
            ),
        ),
        arrow_paths=_validated_promoted_arrow_paths(arrow_paths),
        artifact_dir=tmp_path,
        artifact=FakeArtifact(),
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=None,
    )

    raw_kwargs = captured["raw_arrow_kwargs"]
    assert raw_kwargs["query_signature_ids"] == ["query"]
    raw_plan_bundle = raw_kwargs["raw_plan_bundle"]
    assert isinstance(raw_plan_bundle, RawArrowPlanBundle)
    assert raw_plan_bundle.query_signature_ids == ("query",)
    assert raw_kwargs["rust_featurizer"] is not None
    assert raw_kwargs["arrow_paths"]["cluster_seeds"] == str(cluster_seeds_path)
    assert raw_kwargs["arrow_paths"]["cluster_seed_disallows"] == str(disallow_path)
    assert raw_kwargs["arrow_paths"]["altered_cluster_signatures"] == str(altered_path)
    assert captured["altered_from_arrow"] == ["seed"]
    assert captured["finish_arrow_paths"]["cluster_seeds"] == str(cluster_seeds_path)
    assert captured["planner_inits"] == [()]
    assert captured["planner_plans"] == [("query",)]
    assert result["clusters"] == {"c_seed": ["seed", "query"]}
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["arrow_promoted_incremental"] == 1
    assert telemetry["seed_arrow_reused_source"] == 1
    assert telemetry["raw_arrow_reusable_planner_enabled"] == 1


def test_promoted_linker_replans_window_when_post_featurizer_ram_limit_shrinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(
            retrieval_top_k=25,
            feature_columns=_PROMOTED_TEST_FEATURE_COLUMNS,
        )

    query_ids = [f"q{index:02d}" for index in range(20)]
    captured: dict[str, Any] = {"featurizer_windows": [], "scored_batches": []}

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {"normalization_version": NORMALIZATION_VERSION}
        _last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "python"}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, *_args: object, **_kwargs: object):
            return {"c_seed": ["seed", *query_ids]}

    def fake_limits(**kwargs: object):
        query_count = int(kwargs["query_count"])
        safe_size = 2 if captured["featurizer_windows"] else 10
        return _mock_promoted_limits(
            query_count=query_count,
            query_batch_size=min(query_count, safe_size),
        )

    def fake_featurizer(*_args: object, **kwargs: object) -> object:
        signature_ids = tuple(cast(Any, kwargs["signature_ids"]))
        captured["featurizer_windows"].append(signature_ids)
        return object()

    def fake_linker(*_args: object, **kwargs: object) -> SimpleNamespace:
        batch = tuple(cast(Any, kwargs["query_signature_ids"]))
        captured["scored_batches"].append(batch)
        return SimpleNamespace(
            linked_signature_clusters={signature_id: "c_seed" for signature_id in batch},
            telemetry={"query_count": len(batch), "candidate_row_count": len(batch), "pair_count": len(batch)},
        )

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        fake_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(production_module.feature_port, "build_rust_featurizer_from_arrow_paths", fake_featurizer)

    result = production_module.predict_incremental_promoted_linker_from_arrow_paths(
        FakeClusterer(),
        ["seed", *query_ids],
        cast(
            ANDData,
            SimpleNamespace(
                name_tuples=set(),
                cluster_seeds_disallow=set(),
                altered_cluster_signatures=None,
            ),
        ),
        arrow_paths=_validated_promoted_arrow_paths(_minimal_arrow_paths(tmp_path)),
        artifact_dir=tmp_path,
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=100_000,
        batching_threshold=10,
    )

    assert [len(window) for window in captured["planner_plans"]] == [20, 8, 8, 4]
    assert [len(window) for window in captured["featurizer_windows"]] == [20, 8, 8, 4]
    assert len(captured["scored_batches"]) == 10
    assert {len(batch) for batch in captured["scored_batches"]} == {2}
    assert result["incremental_linker_telemetry"]["raw_arrow_window_memory_replan_count"] == 1


def test_predict_incremental_arrow_promoted_linker_fails_closed_when_single_query_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(
            retrieval_top_k=25,
            feature_columns=_PROMOTED_TEST_FEATURE_COLUMNS,
        )

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = _PROMOTED_TEST_FEATURIZER_INFO
        feature_contract = {"normalization_version": NORMALIZATION_VERSION}
        _last_incremental_seed_setup_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}

    raw_calls: list[object] = []
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **_kwargs: _mock_promoted_limits(single_query_exceeds_budget=True),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        lambda *args, **kwargs: raw_calls.append((args, kwargs)),
    )

    with pytest.raises(MemoryError, match="cannot fit a single query"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            FakeClusterer(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples=set())),
            arrow_paths=_validated_promoted_arrow_paths(_minimal_arrow_paths(tmp_path)),
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=100_000,
            batching_threshold=None,
        )

    assert raw_calls == []


def test_predict_from_arrow_paths_requires_name_counts_index_for_name_count_features(tmp_path: Path) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=object(),
        n_jobs=1,
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)

    with pytest.raises(model_module.MissingArrowArtifactError, match="missing mapping keys: name_counts_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            arrow_paths,
        )

    with pytest.raises(model_module.MissingArrowArtifactError, match="name_counts_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            {
                **arrow_paths,
                "name_counts_index": "missing_name_counts_index",
            },
        )


def test_predict_from_arrow_paths_loads_name_counts_by_default_for_name_count_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=object(),
        n_jobs=1,
    )
    name_counts_index, _metrics = write_name_counts_index(
        tmp_path, tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    arrow_paths = _minimal_arrow_paths(tmp_path)
    arrow_paths["name_counts_index"] = name_counts_index
    write_test_arrow_artifact_manifest(tmp_path, arrow_paths)
    captured: dict[str, Any] = {}

    def fake_build_rust_featurizer_from_arrow_paths(*_args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer_from_arrow_paths
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_from_rust_featurizer",
        lambda *_args, **_kwargs: ({"block": ["s1"]}, None),
    )

    clusterer.predict_from_arrow_paths(
        {"block": ["s1"]},
        arrow_paths,
    )

    assert captured["load_name_counts"] is True


def test_predict_from_arrow_paths_passes_disallow_sidecar_to_rust_featurizer_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        classifier=object(),
        n_jobs=1,
    )
    arrow_paths = _minimal_arrow_paths(tmp_path)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", lambda *_args, **_kwargs: object())

    def fake_predict_from_rust_featurizer(_self: Clusterer, *_args: Any, **kwargs: Any):
        captured.update(kwargs)
        return {"block": ["s1", "s2"]}, None

    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)
    partial_supervision = {("s1", "s2"): 7.0}

    clusterer.predict_from_arrow_paths(
        {"block": ["s1", "s2"]},
        arrow_paths,
        partial_supervision=partial_supervision,
        cluster_seeds_disallow={("s1", "s2")},
    )

    assert captured["partial_supervision"] == partial_supervision
    assert captured["cluster_seeds_disallow"] == {("s1", "s2")}


def test_raw_arrow_runtime_requires_name_counts_index_for_name_count_features() -> None:
    clusterer = SimpleNamespace(featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]))

    with pytest.raises(ValueError, match="requires name_counts_index"):
        production_module.runtime_module.require_arrow_name_counts_index_for_clusterer(
            clusterer,
            {},
            context="Raw Arrow scoring",
        )

    with pytest.raises(model_module.MissingArrowArtifactError, match="name_counts_index"):
        production_module.runtime_module.require_arrow_name_counts_index_for_clusterer(
            clusterer,
            {"name_counts_index": "missing_name_counts_index"},
            context="Raw Arrow scoring",
        )


def test_promoted_incremental_batch_telemetry_does_not_sum_absolute_memory_fields() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 40,
                "memory_stage_budget_bytes": 20,
            },
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 35,
                "memory_stage_budget_bytes": 20,
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


def test_promoted_incremental_batch_telemetry_does_not_sum_raw_plan_seed_counts() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10, "raw_arrow_plan_cluster_count": 2},
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10, "raw_arrow_plan_cluster_count": 2},
        ],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 2
    assert merged["raw_arrow_plan_seed_signature_count"] == 10
    assert merged["raw_arrow_plan_cluster_count"] == 2


def test_promoted_incremental_batch_telemetry_records_conflicting_constant_fields() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10},
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 11},
        ],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["raw_arrow_plan_seed_signature_count"] == 10
    assert merged["raw_arrow_plan_seed_signature_count_batch_conflict_count"] == 1


def test_promoted_incremental_batch_telemetry_keeps_first_after_mixed_type_conflict() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {"query_count": 1, "custom_metric": "unregistered"},
            {"query_count": 1, "custom_metric": 3},
            {"query_count": 1, "custom_metric": 4},
        ],
        batch_sizes=[1, 1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 3
    assert merged["custom_metric"] == "unregistered"
    assert merged["custom_metric_batch_conflict_count"] == 2


def test_promoted_incremental_batch_telemetry_does_not_guess_unknown_numeric_semantics() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [{"query_count": 1, "new_metric": 3}, {"query_count": 1, "new_metric": 4}],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 2
    assert merged["new_metric"] == 3
    assert merged["new_metric_batch_conflict_count"] == 1


def test_raw_window_plan_telemetry_marks_conflicting_string_values() -> None:
    merged: dict[str, int | float | str] = {}

    production_module._merge_raw_window_plan_telemetry(
        merged,
        {
            "raw_arrow_window_plan_query_view": "full",
            "raw_arrow_window_plan_signature_count": 3,
        },
    )
    production_module._merge_raw_window_plan_telemetry(
        merged,
        {
            "raw_arrow_window_plan_query_view": "initial_only",
            "raw_arrow_window_plan_signature_count": 4,
        },
    )

    assert merged["raw_arrow_window_plan_query_view"] == "__mixed__"
    assert merged["raw_arrow_window_plan_signature_count"] == 7.0


def test_predict_incremental_dont_use_cluster_seeds_flag(clusterer_dataset_factory):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_alias")
    block = {"block": ["3", "4", "5", "6"]}

    expected_clusters, _ = clusterer.predict(block, dataset)
    explicit_default_clusters, _ = clusterer.predict(
        block,
        dataset,
        incremental_dont_use_cluster_seeds=False,
    )

    assert _same_partition(expected_clusters, explicit_default_clusters)


def _mock_promoted_limits(
    *,
    query_count: int = 1,
    query_batch_size: int = 1,
    single_query_exceeds_budget: bool = False,
    operational_estimate_source: str = "top_k_largest_components",
    predicted_peak_delta_bytes: int = 2_000,
    predicted_peak_rss_bytes: int = 3_000,
    predicted_pairs_per_batch: int = 40,
    predicted_candidate_rows_per_batch: int = 10,
    pair_chunk_pairs: int = 100,
    pair_chunk_count: int = 1,
) -> model_module.memory_budget.PromotedPhaseALimits:
    return model_module.memory_budget.PromotedPhaseALimits(
        total_ram_bytes=100_000,
        total_ram_source="test",
        current_rss_bytes=1_000,
        current_rss_source="rss:test",
        available_bytes=90_000,
        effective_available_fraction=0.9,
        safety_margin_bytes=1_000,
        stage_budget_fraction=0.5,
        stage_budget_bytes=10_000,
        query_count=int(query_count),
        max_query_batch_size=max(1, int(query_count)),
        query_batch_size=int(query_batch_size),
        component_count=4,
        retrieval_top_k=25,
        candidate_rows_per_query=2,
        conservative_pairs_per_query=4,
        hard_query_batch_size=int(query_batch_size),
        observed_query_count=0,
        observed_candidate_rows_per_query=0,
        observed_pairs_per_query=0,
        observed_safety_multiplier=2.0,
        operational_candidate_rows_per_query=2,
        operational_pairs_per_query=4,
        operational_estimate_source=operational_estimate_source,
        max_component_size=2,
        predicted_candidate_rows_per_batch=int(predicted_candidate_rows_per_batch),
        predicted_pairs_per_batch=int(predicted_pairs_per_batch),
        hard_predicted_candidate_rows_per_batch=int(predicted_candidate_rows_per_batch),
        hard_predicted_pairs_per_batch=int(predicted_pairs_per_batch),
        retrieval_pair_bytes=16,
        retrieval_row_bytes=512,
        pair_label_bytes=8,
        distance_row_bytes=96,
        final_matrix_feature_count=53,
        pairwise_matrix_feature_count=35,
        aggregate_feature_count=18,
        fixed_overhead_bytes=16 * (1 << 20),
        predicted_retrieval_pair_arrays_bytes=0,
        predicted_retrieval_row_bytes=0,
        predicted_pair_label_bytes=0,
        predicted_aggregate_bytes=0,
        predicted_distance_row_bytes=0,
        predicted_final_matrix_bytes=0,
        predicted_fixed_overhead_bytes=16 * (1 << 20),
        predicted_persistent_bytes=0,
        predicted_pair_chunk_bytes=0,
        predicted_peak_delta_bytes=int(predicted_peak_delta_bytes),
        predicted_peak_rss_bytes=int(predicted_peak_rss_bytes),
        pair_chunk_pairs=int(pair_chunk_pairs),
        pair_chunk_count=int(pair_chunk_count),
        pair_chunk_stage_budget_bytes=10_000,
        single_query_predicted_persistent_bytes=100 if not single_query_exceeds_budget else 20_000,
        single_query_exceeds_budget=bool(single_query_exceeds_budget),
    )


def _build_minimal_incremental_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=object(),
        n_jobs=1,
        use_cache=False,
    )


def _strict_rust_context(run_id: str = "test-strict-rust") -> SimpleNamespace:
    return SimpleNamespace(
        operation="cluster_predict",
        backend="rust",
        run_id=run_id,
    )


def test_subblocked_single_letter_cleanup_skips_missing_dataset_rust_featurizer(monkeypatch) -> None:
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="transient_arrow_only",
            cluster_seeds_require={"seed": "seed_cluster"},
            cluster_seeds_disallow=set(),
            name_tuples="filtered",
        ),
    )

    def fake_predict_incremental(self, block_signatures, *_args, **_kwargs):
        del self
        return {"clusters": {"single": list(block_signatures)}}

    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)
    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dataset featurizer sync should be skipped")),
    )

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
    assert _same_partition(
        batch_output, baseline
    ), f"Batch-constraint and baseline partitions differ:\n  batch={batch_output}\n  baseline={baseline}"
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


def test_graph_subblocking_falls_back_to_legacy_specter_for_missing_files(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_io_legacy_fallback")
    captured: dict[str, Any] = {}

    class FailingFallback:
        load_seconds = 0.0
        load_metrics = {}
        stats: list[dict[str, Any]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise FileNotFoundError("missing graph sidecar")

    def fake_factory(*, config):
        del config
        return FailingFallback()

    def fake_legacy(signature_ids, anddata, target_subblock_size=10000, **kwargs):
        del anddata, target_subblock_size, kwargs
        captured["legacy_signature_ids"] = list(signature_ids)
        return {"legacy": list(signature_ids)}

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(model_module, "cluster_with_specter", fake_legacy)

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "5"])
    assert fallback is not None

    output = fallback(["3", "4", "5"], dataset, target_subblock_size=2)

    assert output == {"legacy": ["3", "4", "5"]}
    assert captured["legacy_signature_ids"] == ["3", "4", "5"]
    assert fallback.legacy_fallback_invocation_count == 1
    assert fallback.graph_fallback_errors == [
        {
            "stage": "call",
            "type": "FileNotFoundError",
            "message": "missing graph sidecar",
            "signature_count": 3,
        }
    ]


def test_arrow_graph_subblocking_io_errors_do_not_fall_back_to_legacy(clusterer_dataset_factory, monkeypatch):
    _clusterer, dataset = clusterer_dataset_factory(name="dummy_arrow_graph_io_strict")

    class FailingFallback:
        load_seconds = 0.0
        load_metrics = {}
        stats: list[dict[str, Any]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise OSError("arrow mmap failed")

    monkeypatch.setattr(
        model_module,
        "cluster_with_specter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Arrow graph failures should not use legacy fallback")
        ),
    )

    fallback = model_module._GraphSubblockingFallbackWithLegacyFallback(FailingFallback(), source="arrow")

    with pytest.raises(OSError, match="arrow mmap failed"):
        fallback(["3", "4", "5"], dataset, target_subblock_size=2)

    assert fallback.legacy_fallback_invocation_count == 0
    assert fallback.graph_fallback_errors == []


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

    def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
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
            name_tuples="filtered",
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


def test_finish_incremental_with_seed_links_reclusters_abstains_from_arrow_paths():
    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow residual Phase B should not call legacy predict_helper")

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_paths"] = dict(arrow_paths)
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fail_predict_helper)
    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
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
    arrow_paths = {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"}
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
        arrow_paths=arrow_paths,
    )

    assert result == {"7": ["seed0", "u1"], "8": ["seed1"], "9": ["u2", "u3"]}
    assert captured["block_dict"] == {"block": ["u2", "u3"]}
    assert captured["arrow_paths"] == arrow_paths
    assert captured["partial_supervision"] == {("u2", "u3"): LARGE_DISTANCE}
    assert captured["runtime_context"] is runtime_context
    assert captured["total_ram_bytes"] == 123_456


def test_finish_incremental_with_seed_links_splits_residual_phase_b_by_first_initial():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths, kwargs
        residual_block = list(block_dict["block"])
        residual_blocks.append(residual_block)
        return {"residual_cluster": residual_block}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
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
        arrow_paths={"signatures": "signatures.arrow"},
    )

    assert result == {"7": ["seed"], "8": ["u_a1", "u_a2"], "9": ["u_b1", "u_b2"]}
    assert residual_blocks == [["u_a1", "u_a2"], ["u_b1", "u_b2"]]
    assert clusterer._last_incremental_residual_phase_b_telemetry == {
        "residual_phase_b_signature_count": 4,
        "residual_phase_b_group_count": 2,
        "residual_phase_b_pair_count_before": 6,
        "residual_phase_b_pair_count_after": 2,
        "residual_phase_b_pair_count_saved": 4,
    }


def test_finish_incremental_with_seed_links_residual_phase_b_preserves_same_orcid_group():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []

    def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
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
    )

    assert result == {"7": ["seed"], "8": ["u_a", "u_b"]}
    assert residual_blocks == [["u_a", "u_b"]]
    assert clusterer._last_incremental_residual_phase_b_telemetry["residual_phase_b_group_count"] == 1
    assert clusterer._last_incremental_residual_phase_b_telemetry["residual_phase_b_pair_count_saved"] == 0


def test_build_incremental_seed_setup_uses_arrow_paths_for_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow altered-profile pre-splitting should not call legacy predict_helper")

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_paths"] = dict(arrow_paths)
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
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
    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8", "seed3": "8", "seed4": "9"},
            cluster_seeds_disallow={("seed0", "seed1"), ("seed2", "seed3"), ("seed3", "seed4")},
            altered_cluster_signatures=["seed0", "seed2", "seed4"],
            name_tuples="filtered",
        ),
    )
    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "cluster_seeds": "cluster_seeds.arrow",
    }
    runtime_context = cast(Any, object())

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=runtime_context,
            total_ram_bytes=123_456,
            arrow_paths=arrow_paths,
        )
    )

    assert captured["block_dict"] == {
        "altered_profile_0": ["seed0", "seed1"],
        "altered_profile_1": ["seed2", "seed3"],
    }
    assert captured["arrow_paths"] == {
        "signatures": str(Path("signatures.arrow").resolve()),
        "papers": str(Path("papers.arrow").resolve()),
        "paper_authors": str(Path("paper_authors.arrow").resolve()),
    }
    assert captured["partial_supervision"] == {
        ("seed0", "seed1"): LARGE_DISTANCE,
        ("seed2", "seed3"): LARGE_DISTANCE,
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


def test_build_incremental_seed_setup_loads_altered_signatures_from_arrow_path(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths
        captured["block_dict"] = dict(block_dict)
        captured["partial_supervision"] = dict(kwargs.get("partial_supervision", {}))
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )
    )

    assert captured["block_dict"] == {"altered_profile_0": ["seed0", "seed1"]}
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_signature_count"] == 1


def test_build_incremental_seed_setup_loads_seed_and_altered_signatures_from_arrow_paths(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths
        captured["block_dict"] = dict(block_dict)
        captured["partial_supervision"] = dict(kwargs.get("partial_supervision", {}))
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    seed_table = pa.table(
        {
            "signature_id": pa.array(["seed0", "seed1", "seed2"], type=pa.string()),
            "cluster_id": pa.array(["7", "7", "8"], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, seed_table.schema) as writer:
            writer.write_table(seed_table)
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    altered_table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, altered_table.schema) as writer:
            writer.write_table(altered_table)
    disallow_path = tmp_path / "cluster_seed_disallows.arrow"
    disallow_table = pa.table(
        {
            "signature_id_1": pa.array(["seed0"], type=pa.string()),
            "signature_id_2": pa.array(["seed1"], type=pa.string()),
        }
    )
    with pa.OSFile(str(disallow_path), "wb") as sink:
        with pa.ipc.new_file(sink, disallow_table.schema) as writer:
            writer.write_table(disallow_table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={
                "cluster_seeds": str(cluster_seeds_path),
                "cluster_seed_disallows": str(disallow_path),
                "altered_cluster_signatures": str(altered_path),
            },
        )
    )

    assert captured["block_dict"] == {"altered_profile_0": ["seed0", "seed1"]}
    assert captured["partial_supervision"] == {("seed0", "seed1"): LARGE_DISTANCE}
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2"]}


def test_cluster_seeds_arrow_read_cache_reuses_parse_and_returns_copy(monkeypatch, tmp_path: Path):
    import pyarrow as pa

    model_module._CLUSTER_SEEDS_ARROW_CACHE.clear()
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"seed0": "7", "seed1": "7"})

    open_file_call_count = 0
    original_open_file = pa.ipc.open_file

    def counting_open_file(*args, **kwargs):
        nonlocal open_file_call_count
        open_file_call_count += 1
        return original_open_file(*args, **kwargs)

    monkeypatch.setattr(pa.ipc, "open_file", counting_open_file)

    def fail_read_bytes(self: Path):
        raise AssertionError(f"cache fingerprint should not read Arrow bytes: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    first = model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})
    first["seed0"] = "mutated"
    second = model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})

    assert second == {"seed0": "7", "seed1": "7"}
    assert open_file_call_count == 1


def test_arrow_paths_need_current_cluster_seeds_rewrites_missing_seed_sidecar(tmp_path: Path):
    dataset = SimpleNamespace(cluster_seeds_require={"seed0": "7"})

    assert model_module._arrow_paths_need_current_cluster_seeds(dataset, {}) is True
    assert (
        model_module._arrow_paths_need_current_cluster_seeds(
            dataset,
            {"cluster_seeds": str(tmp_path / "missing_cluster_seeds.arrow")},
        )
        is True
    )


def test_arrow_paths_need_current_cluster_seeds_rewrites_stale_sidecar_when_current_seeds_empty(tmp_path: Path):
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"seed0": "7"})
    dataset = SimpleNamespace(cluster_seeds_require={})

    assert (
        model_module._arrow_paths_need_current_cluster_seeds(
            dataset,
            {"cluster_seeds": str(cluster_seeds_path)},
        )
        is True
    )


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


def test_build_incremental_seed_setup_empty_altered_list_overrides_arrow_path(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    clusterer.predict_from_arrow_paths = cast(
        Any,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty Python altered list is authoritative")),
    )
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=[],
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )
    )

    assert cluster_seeds_require == {"seed0": "7", "seed1": "7", "seed2": "8"}
    assert recluster_map == {}
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_signature_count"] == 0


def test_build_incremental_seed_setup_rejects_arrow_altered_signature_missing_seed(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    altered_table = pa.table({"signature_id": pa.array(["missing_seed"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, altered_table.schema) as writer:
            writer.write_table(altered_table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    with pytest.raises(ValueError, match="must all be present in cluster_seeds_require"):
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )


def test_build_incremental_seed_setup_rejects_missing_declared_altered_arrow_path(tmp_path: Path):
    clusterer = _build_minimal_incremental_clusterer()
    missing_path = tmp_path / "missing_altered_cluster_signatures.arrow"
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(missing_path)},
        )

    assert exc_info.value.missing_files == {"altered_cluster_signatures": str(missing_path)}


def test_top1_consensus_broadcast_only_applies_when_cluster_members_agree():
    def _run(
        mode: Literal["always", "never", "top1_consensus"],
        signature_dists: dict[str, dict[int, tuple[float, int, float]]],
    ) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()
        clusterer.incremental_precluster_broadcast_mode = mode

        def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
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

        def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
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
