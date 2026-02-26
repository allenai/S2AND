import unittest
import pytest
import numpy as np
import pickle
from typing import Any, Dict, List, Union

import s2and.model as model_module
from s2and.data import ANDData
from s2and.model import Clusterer
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.consts import LARGE_DISTANCE
import lightgbm as lgb


def _same_partition(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> bool:
    """Check that two cluster dicts encode the same partition (same groupings, ignoring cluster IDs)."""

    def _to_partition(clusters: Dict[str, List[str]]) -> frozenset:
        return frozenset(frozenset(sigs) for sigs in clusters.values() if sigs)

    return _to_partition(a) == _to_partition(b)


def _clusters(result: Dict[str, Any]) -> Dict[str, List[str]]:
    return dict(result["clusters"])


def _seeds_preserved(clusters: Dict[str, List[str]], seed_groups: List[List[str]]) -> bool:
    """Each seed group must be entirely contained in one predicted cluster."""
    cluster_sets = [set(sigs) for sigs in clusters.values() if sigs]
    for group in seed_groups:
        group_set = set(group)
        if not any(group_set.issubset(cluster_set) for cluster_set in cluster_sets):
            return False
    return True


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            cluster_seeds={"6": {"7": "require"}, "3": {"4": "require"}},
            name="dummy",
            load_name_counts=True,
        )

        features_to_use = [
            "year_diff",
            "misc_features",
        ]
        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 6))
        y_random = np.random.randint(0, 6, 10)
        self.dummy_clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

    def test_predict_incremental(self):
        # base clustering of the random model would be
        # {'0': ['0', '1', '2'], '1': ['3', '4', '5', '8'], '2': ['6', '7']}

        block = ["3", "4", "5", "6", "7", "8"]

        # Non-subblocked (monolithic) is the reference output.
        output_monolithic = _clusters(
            self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=None)
        )
        expected_output = {"0": ["6", "7"], "1": ["3", "4", "5", "8"]}
        assert output_monolithic == expected_output

        # Subblocked output covers all signatures and preserves seed pairs.
        # Note: subblocked and monolithic may differ because the unassigned
        # re-clustering step within the helper operates on subblock-local
        # unassigned sets. This is inherent to subblocking. The frozen-seed
        # approach ensures each subblock sees the same original seeds.
        output_subblocked = _clusters(
            self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=3)
        )
        subblocked_sigs = {s for sigs in output_subblocked.values() for s in sigs}
        assert subblocked_sigs == set(block), f"Subblocked output missing signatures: {set(block) - subblocked_sigs}"
        # Seed pairs must stay together: (3,4) in one cluster, (6,7) in another.
        assert _seeds_preserved(output_subblocked, [["3", "4"], ["6", "7"]])

        self.dummy_dataset.cluster_seeds_disallow = {("5", "7"), ("8", "4"), ("5", "4"), ("8", "7")}
        output = _clusters(self.dummy_clusterer.predict_incremental(block, self.dummy_dataset))
        expected_output = {"0": ["6", "7"], "1": ["3", "4"], "2": ["5", "8"]}
        assert output == expected_output

        self.dummy_dataset.altered_cluster_signatures = ["1", "5"]
        self.dummy_dataset.cluster_seeds_require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
        block = ["3", "4", "8"]
        output = _clusters(
            self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=None)
        )
        expected_output = {"0": ["1", "2", "5", "8"], "1": ["6", "7", "3", "4"]}
        assert output == expected_output


def _build_dummy_clusterer_and_dataset():
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        cluster_seeds={"6": {"7": "require"}, "3": {"4": "require"}},
        name="dummy_chunked",
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


def _mock_incremental_limits(
    chunk_pairs: int,
    accumulator_budget_bytes: int,
    chunk_budget_bytes: int = 4 * 1024 * 1024 * 1024,
) -> Dict[str, Union[int, str]]:
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

    # Simulate the singleton recluster append loop in predict_incremental_helper.
    for signatures in (["new_a"], ["new_b"]):
        cluster_id = model_module._next_unused_cluster_id(pred_clusters, start)
        pred_clusters[str(cluster_id)] = signatures
        start = cluster_id + 1

    assert pred_clusters["2"] == ["existing_singleton_cluster"]
    assert pred_clusters["3"] == ["new_a"]
    assert pred_clusters["4"] == ["new_b"]


def test_predict_incremental_without_seeds_covers_all_signatures():
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    dataset.cluster_seeds_require = {}
    block = ["3", "4", "5", "6", "7", "8"]

    output_no_subblock = _clusters(clusterer.predict_incremental(block, dataset, batching_threshold=None))
    assigned_no_subblock = {signature for signatures in output_no_subblock.values() for signature in signatures}
    assert assigned_no_subblock == set(block)

    # Re-create to get fresh state (dataset.cluster_seeds_require was mutated above).
    clusterer2, dataset2 = _build_dummy_clusterer_and_dataset()
    dataset2.cluster_seeds_require = {}
    output_subblock = _clusters(clusterer2.predict_incremental(block, dataset2, batching_threshold=3))
    assigned_subblock = {signature for signatures in output_subblock.values() for signature in signatures}
    assert assigned_subblock == set(block)

    # Subblocked and non-subblocked should produce the same partition.
    assert _same_partition(output_subblock, output_no_subblock), (
        f"Subblocked and monolithic partitions differ (no seeds):\n"
        f"  subblocked={output_subblock}\n  monolithic={output_no_subblock}"
    )


def test_predict_incremental_phase_split_parity(monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]

    baseline_clusterer, baseline_dataset = _build_dummy_clusterer_and_dataset()
    baseline = _clusters(baseline_clusterer.predict_incremental(block, baseline_dataset, batching_threshold=None))

    phase_split_clusterer, phase_split_dataset = _build_dummy_clusterer_and_dataset()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000),
    )
    phase_split = _clusters(phase_split_clusterer.predict_incremental(block, phase_split_dataset, batching_threshold=3))
    assert _same_partition(
        phase_split, baseline
    ), f"Phase-split and monolithic partitions differ:\n  phase_split={phase_split}\n  monolithic={baseline}"


def test_predict_incremental_phase_split_budget_guard(monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1, chunk_budget_bytes=1),
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=3)
    assert result["phase_b_mode"] == "subblock_local"
    assert int(result["phase_b_required_bytes"]) > int(result["phase_b_budget_bytes"])


def test_predict_incremental_phase_split_budget_approx_fallback(monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1, chunk_budget_bytes=1),
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=3)
    output = _clusters(result)
    assigned = {signature for signatures in output.values() for signature in signatures}
    assert assigned == set(block)
    assert _seeds_preserved(output, [["3", "4"], ["6", "7"]])
    assert result["phase_b_mode"] == "subblock_local"


def test_phase_b_telemetry_exact_vs_subblock(monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer_exact, dataset_exact = _build_dummy_clusterer_and_dataset()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1_000_000),
    )
    exact = clusterer_exact.predict_incremental(block, dataset_exact, batching_threshold=3)
    assert exact["phase_b_mode"] == "exact"
    assert int(exact["phase_b_required_bytes"]) <= int(exact["phase_b_budget_bytes"])

    clusterer_fallback, dataset_fallback = _build_dummy_clusterer_and_dataset()
    monkeypatch.setattr(
        model_module,
        "_compute_incremental_memory_limits",
        lambda *_args, **_kwargs: _mock_incremental_limits(chunk_pairs=3, accumulator_budget_bytes=1, chunk_budget_bytes=1),
    )
    fallback = clusterer_fallback.predict_incremental(block, dataset_fallback, batching_threshold=3)
    assert fallback["phase_b_mode"] == "subblock_local"
    assert int(fallback["phase_b_required_bytes"]) > int(fallback["phase_b_budget_bytes"])


def test_predict_subblocked_processes_subblocks_in_sorted_key_order(monkeypatch):
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    block_signatures = ["3", "4", "5", "6"]
    observed_order: List[str] = []

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
        partial_supervision={},
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        runtime_context=None,
    ):
        del self, dataset, dists, cluster_model_params, partial_supervision
        del use_s2_clusters, incremental_dont_use_cluster_seeds, runtime_context
        key = next(iter(block_dict))
        observed_order.append(key)
        return {f"cluster_{len(observed_order)}": list(block_dict[key])}, None

    monkeypatch.setattr(model_module, "make_subblocks", _fake_make_subblocks)
    monkeypatch.setattr(model_module, "_signature_first_for_rules", lambda _: "john")
    monkeypatch.setattr(Clusterer, "predict_helper", _fake_predict_helper)

    clusterer.predict({"block": block_signatures}, dataset, batching_threshold=3)
    assert observed_order == ["block|subblock=alpha", "block|subblock=zeta"]


def test_predict_helper_fastcluster_single_use_defaults(monkeypatch):
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    captured: Dict[str, object] = {}

    def _capture_cluster_call(
        self, block_signatures, dist_matrix, cluster_model_params, dataset, all_disallow_signature_ids
    ):
        captured["dtype"] = dist_matrix.dtype
        captured["preserve_input"] = dict(cluster_model_params or {}).get("preserve_input")
        return [0 for _ in block_signatures]

    monkeypatch.setattr(Clusterer, "_cluster_one_block", _capture_cluster_call)
    clusterer.predict_helper({"block": ["3", "4"]}, dataset)
    assert captured["dtype"] == np.float64
    assert captured["preserve_input"] is False


def test_predict_helper_fastcluster_precomputed_defaults(monkeypatch):
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    captured: Dict[str, object] = {}

    def _capture_cluster_call(
        self, block_signatures, dist_matrix, cluster_model_params, dataset, all_disallow_signature_ids
    ):
        captured["dtype"] = dist_matrix.dtype
        captured["preserve_input"] = dict(cluster_model_params or {}).get("preserve_input")
        return [0 for _ in block_signatures]

    monkeypatch.setattr(Clusterer, "_cluster_one_block", _capture_cluster_call)
    clusterer.predict_helper({"block": ["3", "4"]}, dataset, dists={"block": np.array([0.5], dtype=np.float16)})
    assert captured["dtype"] == np.float16
    assert captured["preserve_input"] is True


def test_ram_arg_overrides_autodetect(monkeypatch):
    monkeypatch.setattr(model_module, "_detect_cgroup_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    monkeypatch.setattr(model_module, "_detect_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    resolved, source = model_module._resolve_total_ram_bytes_for_incremental(total_ram_bytes=1024)
    assert resolved == 1024
    assert source == "arg"


def test_ram_cgroup_detection_uses_80pct(monkeypatch):
    monkeypatch.setattr(model_module, "_detect_cgroup_total_ram_bytes_best_effort", lambda: (10_000, "cgroup:test"))
    monkeypatch.setattr(model_module, "_detect_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    resolved, source = model_module._resolve_total_ram_bytes_for_incremental()
    assert resolved == 8000
    assert source == "cgroup:test_80pct"


def test_ram_host_detection_uses_80pct(monkeypatch):
    monkeypatch.setattr(model_module, "_detect_cgroup_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    monkeypatch.setattr(model_module, "_detect_total_ram_bytes_best_effort", lambda: (20_000, "psutil.virtual_memory"))
    resolved, source = model_module._resolve_total_ram_bytes_for_incremental()
    assert resolved == 16_000
    assert source == "psutil.virtual_memory_80pct"


def test_ram_detection_failure_raises(monkeypatch):
    monkeypatch.setattr(model_module, "_detect_cgroup_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    monkeypatch.setattr(model_module, "_detect_total_ram_bytes_best_effort", lambda: (None, "unavailable"))
    with pytest.raises(RuntimeError, match="Unable to determine total RAM"):
        model_module._resolve_total_ram_bytes_for_incremental()


def test_ram_arg_precedence_over_cgroup(monkeypatch):
    monkeypatch.setattr(model_module, "_detect_cgroup_total_ram_bytes_best_effort", lambda: (10_000, "cgroup:test"))
    resolved, source = model_module._resolve_total_ram_bytes_for_incremental(total_ram_bytes=4000)
    assert resolved == 4000
    assert source == "arg"


def test_phase_a_memory_prediction_logged_and_bounded(caplog):
    clusterer, dataset = _build_dummy_clusterer_and_dataset()
    block = ["3", "4", "5", "6", "7", "8"]
    rss_now, _ = model_module._current_rss_bytes_best_effort(16 * 1024 * 1024 * 1024)
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
    assert "observed_peak_delta_bytes=" in phase_a_log
    assert "prediction_error_ratio=" in phase_a_log

    ratio_text = phase_a_log.split("prediction_error_ratio=")[1].split()[0]
    ratio = float(ratio_text)
    assert ratio <= 10.0
