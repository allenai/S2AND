from __future__ import annotations

import threading

import numpy as np

import s2and.featurizer as featurizer_mod
from s2and import feature_port, memory_budget
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import build_dummy_dataset


def _mock_chunk_plan(chunk_pairs: int, total_pairs: int) -> dict[str, int | str | float]:
    bytes_per_pair_row = featurizer_mod.NUM_FEATURES * 8 + 128
    predicted_chunk_bytes = int(chunk_pairs) * int(bytes_per_pair_row)
    predicted_features_matrix_bytes = int(total_pairs) * int(featurizer_mod.NUM_FEATURES * 8)
    return {
        "total_ram_bytes": 2 * 1024 * 1024 * 1024,
        "total_ram_source": "test",
        "current_rss_bytes": 256 * 1024 * 1024,
        "current_rss_source": "test",
        "available_bytes": 1024 * 1024 * 1024,
        "safety_margin_bytes": 128 * 1024 * 1024,
        "stage_budget_fraction": 0.25,
        "stage_budget_bytes": 256 * 1024 * 1024,
        "base_chunk_pairs": 10_000,
        "bytes_per_pair_row": int(bytes_per_pair_row),
        "derived_chunk_pairs": int(chunk_pairs),
        "chunk_pairs": int(chunk_pairs),
        "predicted_chunk_bytes": int(predicted_chunk_bytes),
        "predicted_features_matrix_bytes": int(predicted_features_matrix_bytes),
        "predicted_stage_peak_bytes": int(predicted_chunk_bytes + predicted_features_matrix_bytes),
    }


def _build_pairs(count: int) -> list[tuple[str, str, float]]:
    signature_ids = [str(i) for i in range(9)]
    pairs: list[tuple[str, str, float]] = []
    for idx in range(count):
        left = signature_ids[idx % len(signature_ids)]
        right = signature_ids[(idx + 1) % len(signature_ids)]
        if left == right:
            right = signature_ids[(idx + 2) % len(signature_ids)]
        pairs.append((left, right, 0.0))
    return pairs


def test_rust_batch_probe_row_counts_uses_three_probes():
    assert featurizer_mod._rust_batch_probe_row_counts(120_000, probe_count=3, min_total_pairs=30_000) == [
        10_000,
        50_000,
        100_000,
    ]
    derived = featurizer_mod._rust_batch_probe_row_counts(45_000, probe_count=3, min_total_pairs=30_000)
    assert len(derived) == 3
    assert derived[-1] == 45_000
    assert derived[0] < derived[1] < derived[2]


def test_prefault_numpy_pages_inplace_mutates_scratch_buffer():
    scratch = np.full(8193, 7, dtype=np.uint8)
    featurizer_mod._prefault_scratch_array_pages_inplace(scratch)
    assert scratch[0] == 0
    assert scratch[-1] == 0
    assert scratch[4096] == 0
    assert scratch[8192] == 0


def test_rust_batch_plan_never_decreases_fixed_overhead(monkeypatch):
    dataset = build_dummy_dataset("dummy_rust_chunking_calibrated_fixed", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = _build_pairs(5)
    captured_fixed: list[int] = []

    class FakeRustFeaturizer:
        def featurize_pairs(self, pairs, num_threads=None):
            del num_threads
            return [[0.0] * featurizer_mod.NUM_FEATURES for _ in pairs]

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: FakeRustFeaturizer(),
    )
    monkeypatch.setattr(
        featurizer_mod,
        "_maybe_calibrate_rust_batch_fixed_overhead_bytes",
        lambda **_kwargs: 4242,
    )

    def _capturing_plan(**kwargs):
        captured_fixed.append(int(kwargs["fixed_overhead_bytes"]))
        return _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs))

    monkeypatch.setattr(memory_budget, "compute_rust_batch_chunk_plan", _capturing_plan)
    monkeypatch.setattr(
        memory_budget,
        "resolve_rust_batch_prediction_params",
        lambda: {
            "base_chunk_pairs": 10_000,
            "row_overhead_bytes": 128,
            "persistent_row_overhead_bytes": 64,
            "fixed_overhead_bytes": 1_000_000,
        },
    )

    many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert captured_fixed
    assert captured_fixed[0] == 1_000_000


def test_rust_batch_calls_are_chunked_for_progress_updates(monkeypatch):
    dataset = build_dummy_dataset("dummy_rust_chunking", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])

    call_sizes = []

    class FakeRustFeaturizer:
        def featurize_pairs(self, pairs, num_threads=None):
            del num_threads
            call_sizes.append(len(pairs))
            return [[0.0] * featurizer_mod.NUM_FEATURES for _ in pairs]

    fake_rust_featurizer = FakeRustFeaturizer()
    pairs = _build_pairs(5)

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs)),
    )
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: fake_rust_featurizer,
    )

    features, labels, _ = many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert call_sizes == [2, 2, 1]
    assert features.shape[0] == len(pairs)
    assert labels.shape[0] == len(pairs)


def test_rust_batch_prefers_indexed_api_when_available(monkeypatch):
    dataset = build_dummy_dataset("dummy_rust_chunking_indexed", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])

    indexed_call_sizes = []
    selected_indices_seen: list[list[int] | None] = []

    class FakeRustFeaturizer:
        def signature_ids(self):
            return sorted(dataset.signatures.keys())

        def featurize_pairs_matrix_indexed(self, pairs, selected_indices, num_threads, nan_value):
            del num_threads, nan_value
            indexed_call_sizes.append(len(pairs))
            selected_indices_seen.append(selected_indices)
            if selected_indices is None:
                return np.zeros((len(pairs), featurizer_mod.NUM_FEATURES), dtype=np.float64)
            return np.zeros((len(pairs), len(selected_indices)), dtype=np.float64)

        def featurize_pairs_matrix(self, *_args, **_kwargs):
            raise AssertionError("Expected indexed batch API to be used")

    fake_rust_featurizer = FakeRustFeaturizer()
    pairs = _build_pairs(5)

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs)),
    )
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: fake_rust_featurizer,
    )

    features, labels, _ = many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert indexed_call_sizes == [2, 2, 1]
    assert selected_indices_seen
    expected_indices = sorted(
        {
            idx
            for feature_group in featurizer_info.features_to_use
            for idx in featurizer_info.feature_group_to_index[feature_group]
        }
    )
    assert selected_indices_seen[0] == expected_indices
    assert features.shape[0] == len(pairs)
    assert labels.shape[0] == len(pairs)


def test_rust_batch_indexed_api_normalizes_integer_signature_ids(monkeypatch):
    dataset = build_dummy_dataset("dummy_rust_chunking_indexed_int_ids", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    indexed_pairs_seen: list[tuple[int, int]] = []

    class FakeRustFeaturizer:
        def signature_ids(self):
            return sorted(dataset.signatures.keys())

        def featurize_pairs_matrix_indexed(self, pairs, selected_indices, num_threads, nan_value):
            del num_threads, nan_value
            indexed_pairs_seen.extend((int(left), int(right)) for left, right in pairs)
            if selected_indices is None:
                return np.zeros((len(pairs), featurizer_mod.NUM_FEATURES), dtype=np.float64)
            return np.zeros((len(pairs), len(selected_indices)), dtype=np.float64)

    fake_rust_featurizer = FakeRustFeaturizer()
    string_pairs = _build_pairs(5)
    pairs = [(int(left), int(right), label) for left, right, label in string_pairs]

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs)),
    )
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: fake_rust_featurizer,
    )

    features, labels, _ = many_pairs_featurize(
        pairs,  # type: ignore[arg-type]
        dataset,
        featurizer_info,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    signature_index = {sig_id: idx for idx, sig_id in enumerate(sorted(dataset.signatures.keys()))}
    expected_indexed_pairs = [
        (signature_index[str(left)], signature_index[str(right)]) for left, right, _label in pairs
    ]

    assert indexed_pairs_seen == expected_indexed_pairs
    assert features.shape[0] == len(pairs)
    assert labels.shape[0] == len(pairs)


def test_rust_batch_forwards_use_cache_flag(monkeypatch):
    dataset = build_dummy_dataset("dummy_rust_chunking_use_cache_forward", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    forwarded_use_cache: list[bool | None] = []

    class FakeRustFeaturizer:
        def featurize_pairs(self, pairs, num_threads=None):
            del num_threads
            return [[0.0] * featurizer_mod.NUM_FEATURES for _ in pairs]

    fake_rust_featurizer = FakeRustFeaturizer()
    pairs = [("0", "1", 0.0)]

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=1, total_pairs=len(pairs)),
    )
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: forwarded_use_cache.append(use_cache) or fake_rust_featurizer,
    )

    many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert forwarded_use_cache
    assert set(forwarded_use_cache) == {False}


def test_rust_batch_prediction_matches_observed_real_workload(monkeypatch, caplog):
    dataset = build_dummy_dataset("dummy_rust_chunking_prediction", load_name_counts=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = _build_pairs(12_000)

    class FakeRustFeaturizer:
        def featurize_pairs(self, pairs, num_threads=None):
            del num_threads
            return np.zeros((len(pairs), featurizer_mod.NUM_FEATURES), dtype=np.float64)

    fake_rust_featurizer = FakeRustFeaturizer()
    total_ram_bytes = 2 * 1024 * 1024 * 1024
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=featurizer_mod.NUM_FEATURES,
        total_pairs=len(pairs),
        total_ram_bytes=total_ram_bytes,
    )

    monkeypatch.setattr(featurizer_mod, "_use_rust_featurizer", lambda _rc=None: True)
    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, use_cache=False, **_kw: fake_rust_featurizer,
    )

    stop = threading.Event()
    rss_peak = {"value": 0}
    rss_before, _ = memory_budget.current_rss_bytes_best_effort(int(plan["total_ram_bytes"]))
    rss_peak["value"] = rss_before

    def _sample_peak() -> None:
        while not stop.is_set():
            rss_now, _ = memory_budget.current_rss_bytes_best_effort(int(plan["total_ram_bytes"]))
            if rss_now > rss_peak["value"]:
                rss_peak["value"] = rss_now
            stop.wait(0.005)

    worker = threading.Thread(target=_sample_peak, daemon=True)
    worker.start()
    try:
        with caplog.at_level("INFO", logger="s2and"):
            many_pairs_featurize(
                pairs,
                dataset,
                featurizer_info,
                n_jobs=2,
                use_cache=False,
                chunk_size=100,
                nan_value=np.nan,
                total_ram_bytes=total_ram_bytes,
            )
    finally:
        stop.set()
        worker.join(timeout=2)

    rss_after, _ = memory_budget.current_rss_bytes_best_effort(int(plan["total_ram_bytes"]))
    summary = memory_budget.summarize_prediction_accuracy(
        stage_name="pair_featurization_rust_batch_test",
        predicted_peak_delta_bytes=int(plan["predicted_stage_peak_delta_bytes"]),
        rss_before_bytes=rss_before,
        rss_peak_bytes=int(rss_peak["value"]),
        rss_after_bytes=rss_after,
    )
    telemetry_lines = [
        record.message for record in caplog.records if "Telemetry: pair_featurization_memory" in record.message
    ]
    assert telemetry_lines
    telemetry = telemetry_lines[-1]
    assert "prediction_contract_version=" in telemetry
    assert "predicted_peak_delta_bytes=" in telemetry
    assert "predicted_peak_rss_bytes=" in telemetry
    assert "rss_before_bytes=" in telemetry
    assert "rss_peak_bytes=" in telemetry
    assert "observed_peak_delta_bytes=" in telemetry
    assert float(summary["prediction_error_ratio"]) <= 3.0
