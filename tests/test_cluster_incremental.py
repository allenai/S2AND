import inspect
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.model as model_module
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, IncrementalDistStats


def _same_partition(a: dict[str, list[str]], b: dict[str, list[str]]) -> bool:
    """Check that two cluster dicts encode the same partition (same groupings, ignoring cluster IDs)."""

    def _to_partition(clusters: dict[str, list[str]]) -> frozenset:
        return frozenset(frozenset(sigs) for sigs in clusters.values() if sigs)

    return _to_partition(a) == _to_partition(b)


def _clusters(result: dict[str, Any]) -> dict[str, list[str]]:
    return dict(result["clusters"])


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
        load_name_counts=True,
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(1)
    X_random = rng.random((10, 6))
    y_random = rng.randint(0, 6, 10)
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(X_random, y_random),
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
    output_monolithic = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=None))
    expected_output = {"0": ["6", "7"], "1": ["3", "4", "5", "8"]}
    assert output_monolithic == expected_output

    # Subblocked output covers all signatures and preserves seed pairs.
    # Note: subblocked and monolithic may differ because the unassigned
    # re-clustering step within the helper operates on subblock-local
    # unassigned sets. This is inherent to subblocking. The frozen-seed
    # approach ensures each subblock sees the same original seeds.
    output_subblocked = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=3))
    subblocked_sigs = {s for sigs in output_subblocked.values() for s in sigs}
    assert subblocked_sigs == set(block), f"Subblocked output missing signatures: {set(block) - subblocked_sigs}"
    # Seed pairs must stay together: (3,4) in one cluster, (6,7) in another.
    assert _seeds_preserved(output_subblocked, [["3", "4"], ["6", "7"]])

    dummy_dataset.cluster_seeds_disallow = {("5", "7"), ("8", "4"), ("5", "4"), ("8", "7")}
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset))
    expected_output = {"0": ["6", "7"], "1": ["3", "4"], "2": ["5", "8"]}
    assert output == expected_output

    dummy_dataset.altered_cluster_signatures = ["1", "5"]
    dummy_dataset.cluster_seeds_require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
    block = ["3", "4", "8"]
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=None))
    expected_output = {"0": ["1", "2", "5", "8"], "1": ["6", "7", "3", "4"]}
    assert output == expected_output


def test_predict_incremental_return_contract(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_contract")
    canned = {
        "clusters": {"0": ["3", "4"], "1": ["5", "6", "7", "8"]},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 123,
        "phase_b_required_bytes": 120,
    }

    def _fake_predict_incremental_helper(self, *args, **kwargs):
        del self, args, kwargs
        return dict(canned)

    monkeypatch.setattr(Clusterer, "_predict_incremental_helper", _fake_predict_incremental_helper)

    payload = clusterer.predict_incremental(block, dataset, batching_threshold=None)
    assert payload == canned

    clusters_only = clusterer.predict_incremental(
        block,
        dataset,
        batching_threshold=None,
        return_clusters_only=True,
    )
    assert clusters_only == canned["clusters"]


def test_predict_incremental_public_signature_has_no_promoted_override_args() -> None:
    signature = inspect.signature(Clusterer.predict_incremental)

    assert "incremental_linker_private" not in signature.parameters
    assert "incremental_linker_artifact_path" not in signature.parameters
    assert "incremental_linker_query_view" not in signature.parameters


def test_predict_incremental_rust_promoted_linker_uses_seed_link_seam(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker")
    block = ["3", "4", "5", "6", "7", "8"]
    residual_blocks: list[list[str]] = []
    residual_total_ram_bytes: list[int | None] = []
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-seed-link-seam",
        source="S2AND_BACKEND",
    )

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context
        residual_blocks.append(list(block_dict["block"]))
        residual_total_ram_bytes.append(total_ram_bytes)
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    monkeypatch.setattr(
        model_module,
        "_resolve_total_ram_bytes_for_incremental",
        lambda _total=None: (1_000_000_000, "test"),
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_000, "rss:test"))
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    artifact = SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25))
    retriever = object()
    captured_inputs: dict[str, Any] = {}
    captured_runtime: dict[str, Any] = {}
    monkeypatch.setattr(artifact_module, "load_incremental_linking_artifact", lambda _path: artifact)

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            retriever=retriever,
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        captured_runtime["clusterer"] = clusterer_arg
        captured_runtime["artifact"] = artifact_arg
        captured_runtime.update(kwargs)
        return SimpleNamespace(
            linked_signature_clusters={"5": "1"},
            telemetry={"query_count": 2, "link_count": 1, "abstain_count": 1},
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    result = clusterer.predict_incremental(
        block,
        dataset,
        batching_threshold=None,
    )

    assert captured_inputs["query_signature_ids"] == ["5", "8"]
    assert captured_runtime["query_signature_ids"] == ["5", "8"]
    assert captured_runtime["queries"] == ("query-5", "query-8")
    assert captured_runtime["retriever"] is retriever
    assert captured_runtime["artifact"] is artifact
    assert captured_runtime["total_ram_bytes"] == 1_000_000_000
    assert captured_runtime["seed_setup"][0] == captured_inputs["cluster_seeds_require"]
    assert residual_blocks == [["8"]]
    assert residual_total_ram_bytes == [1_000_000_000]
    assert any(set(signatures) == {"3", "4", "5"} for signatures in result["clusters"].values())
    assert any(set(signatures) == {"8"} for signatures in result["clusters"].values())
    assert result["incremental_linker_telemetry"]["link_count"] == 1


def test_predict_incremental_explicit_rust_backend_uses_promoted_linker_by_default(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker_default")
    block = ["3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental",
        source="S2AND_BACKEND",
    )
    captured: dict[str, Any] = {}
    promoted_payload = {
        "clusters": {"promoted": list(block)},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)

    def fake_promoted_mode(self, block_signatures, dataset_arg, **kwargs):
        captured["self"] = self
        captured["block_signatures"] = list(block_signatures)
        captured["dataset"] = dataset_arg
        captured.update(kwargs)
        return dict(promoted_payload)

    monkeypatch.setattr(Clusterer, "_predict_incremental_promoted_linker", fake_promoted_mode)

    result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert result == promoted_payload
    assert captured["self"] is clusterer
    assert captured["block_signatures"] == block
    assert captured["dataset"] is dataset
    assert captured["runtime_context"] is runtime_context


def test_predict_incremental_auto_backend_uses_promoted_linker_when_auto_resolves_to_rust(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_linker_default")
    block = ["3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="auto",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-auto-promoted-incremental",
        source="S2AND_BACKEND",
    )
    promoted_payload = {
        "clusters": {"promoted": list(block)},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Clusterer,
        "_predict_incremental_promoted_linker",
        lambda *args, **kwargs: dict(promoted_payload),
    )

    assert clusterer.predict_incremental(block, dataset, batching_threshold=None) == promoted_payload


def test_predict_incremental_promoted_linker_batches_queries(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_batch")
    block = ["3", "4", "5", "6", "7", "8"]
    residual_blocks: list[list[str]] = []
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-batch",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        residual_blocks.append(list(block_dict["block"]))
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    artifact = SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25))
    captured_inputs: dict[str, Any] = {}
    runtime_batches: list[list[str]] = []
    monkeypatch.setattr(artifact_module, "load_incremental_linking_artifact", lambda _path: artifact)

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        batch = [str(signature_id) for signature_id in kwargs["query_signature_ids"]]
        runtime_batches.append(batch)
        return SimpleNamespace(
            linked_signature_clusters={"5": "1"} if batch == ["5"] else {},
            telemetry={
                "query_count": len(batch),
                "candidate_row_count": 10 + len(batch),
                "pair_count": 20 + len(batch),
                "link_count": 1 if batch == ["5"] else 0,
                "abstain_count": 0 if batch == ["5"] else len(batch),
                "retrieval_top_k": 25,
                "seed_signature_count": 4,
                "seed_component_count": 2,
            },
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=1)

    assert captured_inputs["query_signature_ids"] == ["5", "8"]
    assert runtime_batches == [["5"], ["8"]]
    assert residual_blocks == [["8"]]
    assert any(set(signatures) == {"3", "4", "5"} for signatures in result["clusters"].values())
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["query_count"] == 2
    assert telemetry["candidate_row_count"] == 22
    assert telemetry["pair_count"] == 42
    assert telemetry["link_count"] == 1
    assert telemetry["abstain_count"] == 1
    assert telemetry["retrieval_top_k"] == 25
    assert telemetry["seed_signature_count"] == 4
    assert telemetry["seed_component_count"] == 2
    assert telemetry["query_batch_count"] == 2
    assert telemetry["query_batch_size_configured"] == 1
    assert telemetry["query_batch_size_min"] == 1
    assert telemetry["query_batch_size_max"] == 1


def test_predict_incremental_promoted_linker_recalibrates_query_batch_size(
    clusterer_dataset_factory,
    monkeypatch,
    caplog,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_calibration")
    block = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-calibration",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_200, "rss:test"))

    def fake_limits(**kwargs):
        query_count = int(kwargs["query_count"])
        cap = kwargs.get("max_query_batch_size")
        max_batch = query_count if cap is None else min(query_count, int(cap))
        observed = int(kwargs.get("observed_query_count", 0) or 0) > 0
        query_batch_size = 0 if query_count == 0 else max(1, min(max_batch, 3 if observed else 1))
        return {
            "query_batch_size": query_batch_size,
            "single_query_exceeds_budget": False,
            "single_query_predicted_persistent_bytes": 100,
            "stage_budget_bytes": 10_000,
            "total_ram_bytes": 100_000,
            "current_rss_bytes": 1_000,
            "safety_margin_bytes": 1_000,
            "available_bytes": 90_000,
            "predicted_peak_delta_bytes": 2_000 + query_batch_size,
            "predicted_peak_rss_bytes": 3_000 + query_batch_size,
            "operational_estimate_source": "observed_probe" if observed else "top_k_largest_components",
            "predicted_pairs_per_batch": 40 * query_batch_size,
            "predicted_candidate_rows_per_batch": 10 * query_batch_size,
            "pair_chunk_pairs": 100,
            "pair_chunk_count": 1 if query_batch_size else 0,
        }

    monkeypatch.setattr(model_module, "_compute_promoted_incremental_limits", lambda **kwargs: fake_limits(**kwargs))

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25)),
    )

    def fake_build_inputs(**kwargs):
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    runtime_batches: list[list[str]] = []

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        batch = [str(signature_id) for signature_id in kwargs["query_signature_ids"]]
        runtime_batches.append(batch)
        return SimpleNamespace(
            linked_signature_clusters={},
            telemetry={
                "query_count": len(batch),
                "candidate_row_count": 2 * len(batch),
                "pair_count": 4 * len(batch),
                "link_count": 0,
                "abstain_count": len(batch),
                "retrieval_top_k": 25,
                "seed_signature_count": 4,
                "seed_component_count": 2,
            },
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    with caplog.at_level("INFO", logger="s2and"):
        result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert runtime_batches == [["0"], ["1", "2", "5"], ["8"]]
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["query_batch_count"] == 3
    assert telemetry["query_batch_size_max"] == 3
    assert telemetry["memory_initial_query_batch_size"] == 1
    assert telemetry["memory_final_query_batch_size"] == 3
    assert telemetry["memory_observed_calibration_applied"] == 1
    assert telemetry["memory_final_operational_estimate_source"] == "observed_probe"
    assert telemetry["memory_predicted_peak_delta_bytes_max"] == 2003
    assert any(
        record.message.startswith("Telemetry: incremental_promoted_query_batch_calibration ")
        for record in caplog.records
    )


def test_predict_incremental_promoted_linker_fails_closed_when_single_query_exceeds_budget(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_budget_fail")
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-budget-fail",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_module,
        "_get_rust_featurizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should fail before featurizer build")),
    )

    import s2and.incremental_linking.artifact as artifact_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25)),
    )
    monkeypatch.setattr(
        model_module,
        "_compute_promoted_incremental_limits",
        lambda **_kwargs: {
            "single_query_exceeds_budget": True,
            "single_query_predicted_persistent_bytes": 20_000,
            "stage_budget_bytes": 10_000,
            "total_ram_bytes": 100_000,
            "current_rss_bytes": 80_000,
            "safety_margin_bytes": 10_000,
        },
    )

    with pytest.raises(MemoryError, match="cannot fit a single query"):
        clusterer.predict_incremental(["3", "4", "5"], dataset, batching_threshold=None)


def test_predict_incremental_helper_deprecated_shim(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_deprecated_shim")
    block = ["3", "4", "5"]
    canned = {
        "clusters": {"0": ["3", "4", "5"]},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 24,
        "phase_b_required_bytes": 24,
    }

    def _fake_private(self, *args, **kwargs):
        del self, args, kwargs
        return dict(canned)

    monkeypatch.setattr(Clusterer, "_predict_incremental_helper", _fake_private)
    with pytest.deprecated_call(match="predict_incremental_helper"):
        output = clusterer.predict_incremental_helper(block, dataset)
    assert output == canned


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


def test_clusterer_init_prefers_legacy_seed_flag_name():
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(7)
    X_random = rng.random((10, 6))
    y_random = rng.randint(0, 2, 10)

    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=7, data_random_seed=7, feature_fraction_seed=7).fit(X_random, y_random),
        dont_merge_cluster_seeds=False,
        n_jobs=1,
        use_cache=False,
    )

    assert clusterer.dont_merge_cluster_seeds is False


def _mock_incremental_limits(
    chunk_pairs: int,
    accumulator_budget_bytes: int,
    chunk_budget_bytes: int = 4 * 1024 * 1024 * 1024,
) -> dict[str, int | str]:
    return {
        "total_ram_bytes": 16 * 1024 * 1024 * 1024,
        "total_ram_source": "test",
        "current_rss_bytes": 512 * 1024 * 1024,
        "current_rss_source": "test",
        "available_bytes": 8 * 1024 * 1024 * 1024,
        "chunk_budget_bytes": int(chunk_budget_bytes),
        "accumulator_budget_bytes": int(accumulator_budget_bytes),
        "bytes_per_pair": 1024,
        "derived_chunk_pairs": int(chunk_pairs),
        "chunk_pairs": int(chunk_pairs),
        "accumulator_warn": 1_000_000,
        "accumulator_max": 2_000_000,
    }


def _build_minimal_incremental_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=object(),
        n_jobs=1,
        use_cache=False,
    )


def test_batch_phase_a_subblocks_merges_adjacent_small_work_items():
    work_items = [
        model_module._PhaseASubblockWorkItem("alpha", ("u0",), 4),
        model_module._PhaseASubblockWorkItem("beta", ("u1",), 4),
        model_module._PhaseASubblockWorkItem("gamma", ("u2",), 4),
        model_module._PhaseASubblockWorkItem("delta", ("u3", "u4"), 8),
        model_module._PhaseASubblockWorkItem("omega", ("u5", "u6", "u7"), 12),
    ]

    unbatched = model_module._batch_phase_a_subblocks(work_items, batch_target_pairs=None)
    assert [batch.subblock_keys for batch in unbatched] == [
        ("alpha",),
        ("beta",),
        ("gamma",),
        ("delta",),
        ("omega",),
    ]

    batched = model_module._batch_phase_a_subblocks(work_items, batch_target_pairs=10)
    assert [batch.subblock_keys for batch in batched] == [
        ("alpha", "beta", "gamma"),
        ("delta",),
        ("omega",),
    ]
    assert [batch.unassigned_signature_ids for batch in batched] == [
        ("u0", "u1", "u2"),
        ("u3", "u4"),
        ("u5", "u6", "u7"),
    ]
    assert [batch.estimated_pairs for batch in batched] == [12, 8, 12]

    with pytest.raises(ValueError, match="batch_target_pairs must be positive"):
        model_module._batch_phase_a_subblocks(work_items, batch_target_pairs=0)


def test_next_unused_cluster_id_prevents_overwrite():
    pred_clusters = {
        "0": ["s0"],
        "1": ["s1"],
        "2": ["existing_singleton_cluster"],
    }
    start = model_module._next_unused_cluster_id(pred_clusters, 2)
    assert start == 3

    # Simulate the singleton recluster append loop in _predict_incremental_helper.
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

    output_no_subblock = _clusters(clusterer.predict_incremental(block, dataset, batching_threshold=None))
    assigned_no_subblock = {signature for signatures in output_no_subblock.values() for signature in signatures}
    assert assigned_no_subblock == set(block)

    # Re-create to get fresh state (dataset.cluster_seeds_require was mutated above).
    clusterer2, dataset2 = clusterer_dataset_factory()
    dataset2.cluster_seeds_require = {}
    output_subblock = _clusters(clusterer2.predict_incremental(block, dataset2, batching_threshold=3))
    assigned_subblock = {signature for signatures in output_subblock.values() for signature in signatures}
    assert assigned_subblock == set(block)

    # Subblocked and non-subblocked should produce the same partition.
    assert _same_partition(output_subblock, output_no_subblock), (
        f"Subblocked and monolithic partitions differ (no seeds):\n"
        f"  subblocked={output_subblock}\n  monolithic={output_no_subblock}"
    )


def test_predict_incremental_phase_split_parity(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]

    baseline_clusterer, baseline_dataset = clusterer_dataset_factory()
    baseline = _clusters(baseline_clusterer.predict_incremental(block, baseline_dataset, batching_threshold=None))

    phase_split_clusterer, phase_split_dataset = clusterer_dataset_factory()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000),
    )
    phase_split = _clusters(phase_split_clusterer.predict_incremental(block, phase_split_dataset, batching_threshold=3))
    assert _same_partition(
        phase_split, baseline
    ), f"Phase-split and monolithic partitions differ:\n  phase_split={phase_split}\n  monolithic={baseline}"


def test_predict_incremental_batch_constraint_path_parity(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]

    baseline_clusterer, baseline_dataset = clusterer_dataset_factory()
    baseline = _clusters(baseline_clusterer.predict_incremental(block, baseline_dataset, batching_threshold=None))

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

    def _fake_get_constraints_matrix_indexed_rust(dataset, indexed_pairs, **kwargs):
        calls["batch"] += 1
        dont_merge = kwargs.get("dont_merge_cluster_seeds", True)
        incremental_flag = kwargs.get("incremental_dont_use_cluster_seeds", False)
        return [
            dataset.get_constraint(
                sig_ids[i1],
                sig_ids[i2],
                dont_merge_cluster_seeds=dont_merge,
                incremental_dont_use_cluster_seeds=incremental_flag,
            )
            for i1, i2 in indexed_pairs
        ]

    monkeypatch.setattr(model_module, "get_constraints_matrix_indexed_rust", _fake_get_constraints_matrix_indexed_rust)
    monkeypatch.setattr(model_module, "get_constraint_rust", lambda *_args, **_kwargs: None)

    batch_output = _clusters(batch_clusterer.predict_incremental(block, batch_dataset, batching_threshold=None))
    assert _same_partition(
        batch_output, baseline
    ), f"Batch-constraint and baseline partitions differ:\n  batch={batch_output}\n  baseline={baseline}"
    assert calls["batch"] > 0


def test_predict_incremental_phase_split_budget_approx_fallback(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = clusterer_dataset_factory()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(
            chunk_pairs=3,
            accumulator_budget_bytes=1,
            chunk_budget_bytes=1,
        ),
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=3)
    output = _clusters(result)
    assigned = {signature for signatures in output.values() for signature in signatures}
    assert assigned == set(block)
    assert _seeds_preserved(output, [["3", "4"], ["6", "7"]])
    assert result["phase_b_mode"] == "subblock_local"


def test_phase_b_mode_tracks_exact_vs_subblock_budget(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer_exact, dataset_exact = clusterer_dataset_factory()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000),
    )
    exact = clusterer_exact.predict_incremental(block, dataset_exact, batching_threshold=3)
    assert exact["phase_b_mode"] == "exact"
    assert int(exact["phase_b_required_bytes"]) <= int(exact["phase_b_budget_bytes"])

    clusterer_fallback, dataset_fallback = clusterer_dataset_factory()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(
            chunk_pairs=3,
            accumulator_budget_bytes=1,
            chunk_budget_bytes=1,
        ),
    )
    fallback = clusterer_fallback.predict_incremental(block, dataset_fallback, batching_threshold=3)
    assert fallback["phase_b_mode"] == "subblock_local"
    assert int(fallback["phase_b_required_bytes"]) > int(fallback["phase_b_budget_bytes"])


def test_predict_subblocked_processes_subblocks_in_sorted_key_order(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory()
    block_signatures = ["3", "4", "5", "6"]
    observed_order: list[str] = []

    def _fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None):
        del signatures, anddata, maximum_size, first_k_letter_counts_sorted
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


def test_phase_a_batching_preserves_partition(clusterer_dataset_factory, monkeypatch):
    block = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=10, accumulator_budget_bytes=1_000_000),
    )

    def _fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None):
        del signatures, anddata, maximum_size, first_k_letter_counts_sorted
        return {
            "alpha": ["0", "3"],
            "beta": ["1", "4"],
            "gamma": ["2", "6"],
            "delta": ["5", "7", "8"],
        }

    monkeypatch.setattr(model_module, "make_subblocks", _fake_make_subblocks)

    baseline_clusterer, baseline_dataset = clusterer_dataset_factory(name="dummy_phase_a_batching_off")
    baseline = baseline_clusterer.predict_incremental(block, baseline_dataset, batching_threshold=3)

    batched_clusterer, batched_dataset = clusterer_dataset_factory(name="dummy_phase_a_batching_on")
    batched_clusterer.incremental_phase_a_pair_batch_target_multiple = 1.0
    batched = batched_clusterer.predict_incremental(block, batched_dataset, batching_threshold=3)

    baseline_clusters = _clusters(baseline)
    batched_clusters = _clusters(batched)
    assert _same_partition(batched_clusters, baseline_clusters), (
        "Phase A batching changed the predicted partition on the dummy fixture:\n"
        f"  batched={batched_clusters}\n"
        f"  baseline={baseline_clusters}"
    )
    assert _seeds_preserved(batched_clusters, [["3", "4"], ["6", "7"]])


def test_phase_a_overflow_surfaces_in_result(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory()
    block = ["3", "4", "5", "6", "7", "8"]

    limits = _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000)
    limits["accumulator_warn"] = 1
    limits["accumulator_max"] = 1
    monkeypatch.setattr(model_module, "_compute_incremental_memory_limits", lambda *_args, **_kwargs: limits)

    result = clusterer.predict_incremental(block, dataset, batching_threshold=3)

    assert bool(result["phase_a_accumulator_overflow_early_stop"]) is True


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
    assert residual_blocks == [["u2"]]
    assert residual_total_ram_bytes == [123_456]


def test_build_incremental_seed_setup_passes_total_ram_to_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    recluster_blocks: list[list[str]] = []
    recluster_total_ram_bytes: list[int | None] = []

    def fake_predict_helper(
        block_dict,
        dataset,
        *,
        incremental_dont_use_cluster_seeds,
        partial_supervision,
        runtime_context,
        total_ram_bytes=None,
    ):
        del dataset, partial_supervision, runtime_context
        assert incremental_dont_use_cluster_seeds is True
        recluster_blocks.append(list(block_dict["block"]))
        recluster_total_ram_bytes.append(total_ram_bytes)
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        type(
            "IncrementalDataset",
            (),
            {
                "cluster_seeds_require": {"seed0": "7", "seed1": "7", "seed2": "8"},
                "altered_cluster_signatures": ["seed0"],
            },
        )(),
    )

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = clusterer._build_incremental_seed_setup(
        dataset,
        {},
        runtime_context=cast(Any, object()),
        total_ram_bytes=123_456,
    )

    assert recluster_blocks == [["seed0", "seed1"]]
    assert recluster_total_ram_bytes == [123_456]
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2"]}


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
