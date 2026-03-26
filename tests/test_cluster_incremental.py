from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.model as model_module
from s2and import memory_budget
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


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
        classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
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
        classifier=lgb.LGBMClassifier(random_state=7, data_random_seed=7, feature_fraction_seed=7).fit(
            X_random, y_random
        ),
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


def test_phase_b_telemetry_exact_vs_subblock(clusterer_dataset_factory, monkeypatch):
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


def test_phase_a_memory_prediction_logged_and_bounded(clusterer_dataset_factory, caplog):
    clusterer, dataset = clusterer_dataset_factory()
    block = ["3", "4", "5", "6", "7", "8"]
    rss_now, _ = memory_budget.current_rss_bytes_best_effort(16 * 1024 * 1024 * 1024)
    total_ram_bytes = max(8 * 1024 * 1024 * 1024, int(rss_now) + 4 * 1024 * 1024 * 1024)

    with caplog.at_level("INFO", logger="s2and"):
        clusterer.predict_incremental(
            block,
            dataset,
            batching_threshold=3,
            total_ram_bytes=total_ram_bytes,
        )

    phase_a_logs = [record.message for record in caplog.records if "Telemetry: phase_split_phase_a" in record.message]
    assert phase_a_logs, "Expected phase_split_phase_a telemetry log with memory prediction metrics"
    phase_a_log = phase_a_logs[-1]
    assert "prediction_contract_version=" in phase_a_log
    assert "predicted_peak_delta_bytes=" in phase_a_log
    assert "predicted_peak_rss_bytes=" in phase_a_log
    assert "predicted_bytes=" in phase_a_log
    assert "rss_before_bytes=" in phase_a_log
    assert "rss_peak_bytes=" in phase_a_log
    assert "rss_after_bytes=" in phase_a_log
    assert "observed_peak_delta_bytes=" in phase_a_log
    assert "prediction_error_ratio=" in phase_a_log
    assert "underpredicted=" in phase_a_log

    # Keep Phase A modeled terms stable for calibration and post-hoc analysis.
    assert "accumulator_entries_peak_sample=" in phase_a_log
    assert "phase_a_pair_buffer_peak_bytes=" in phase_a_log
    assert "phase_a_pair_buffer_entry_bytes=" in phase_a_log

    ratio_text = phase_a_log.split("prediction_error_ratio=")[1].split()[0]
    ratio = float(ratio_text)
    assert ratio <= 10.0


def test_phase_a_overflow_surfaces_in_result_and_telemetry(clusterer_dataset_factory, monkeypatch, caplog):
    clusterer, dataset = clusterer_dataset_factory()
    block = ["3", "4", "5", "6", "7", "8"]

    limits = _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000)
    limits["accumulator_warn"] = 1
    limits["accumulator_max"] = 1
    monkeypatch.setattr(model_module, "_compute_incremental_memory_limits", lambda *_args, **_kwargs: limits)

    with caplog.at_level("INFO", logger="s2and"):
        result = clusterer.predict_incremental(block, dataset, batching_threshold=3)

    assert bool(result["phase_a_accumulator_overflow_early_stop"]) is True
    overflow_logs = [
        record.message for record in caplog.records if "Telemetry: phase_split_phase_a_overflow" in record.message
    ]
    assert overflow_logs
    assert "overflow_early_stop=True" in overflow_logs[-1]
